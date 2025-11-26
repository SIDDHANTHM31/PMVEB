import Foundation
import AVFoundation

// MARK: - SpeechAlertManager
// All public APIs are safe to call from any thread. Internally we switch to @MainActor
// because AVAudio / AVSpeechSynthesizer is not concurrency-safe.
@MainActor
final class SpeechAlertManager: NSObject {
    static let shared = SpeechAlertManager()

    private let synthesizer = AVSpeechSynthesizer()
    private var preferredVoice: AVSpeechSynthesisVoice?
    private var lastAnnouncedIDs = Set<UUID>()
    private var lastAnnouncementDate: Date?
    private let announcementCooldown: TimeInterval = 30.0 // seconds minimum between announcement bursts
    private let maxCombinedAlerts = 5 // maximum alerts to combine into one utterance

    private override init() {
        super.init()
        synthesizer.delegate = self
        configureAudioSession()
        Task { @MainActor in
            await loadPreferredVoiceSafely()
        }
    }

    // MARK: - Public API

    /// Announce critical alerts. Safe to call from any thread/context.
    func announceCriticalAlerts(_ alerts: [MaintenanceAlert]) {
        Task { @MainActor in
            await self._announceCriticalAlerts(alerts)
        }
    }

    /// Reset announced history (use when user clears alerts or on new session)
    func resetAnnouncedHistory() {
        Task { @MainActor in
            lastAnnouncedIDs.removeAll()
            lastAnnouncementDate = nil
        }
    }

    // Optional: announce a single custom message (safe to call from any thread)
    func announceMessage(_ message: String) {
        Task { @MainActor in
            await self._announceMessage(message)
        }
    }

    // MARK: - Internal (MainActor)

    private func _announceMessage(_ message: String) async {
        // Rate-limit quick repeated manual messages as well
        if let last = lastAnnouncementDate, Date().timeIntervalSince(last) < 1.0 {
            // Too soon for another manual short message
            return
        }
        lastAnnouncementDate = Date()
        let utterance = makeUtterance(from: message)
        speakUtterance(utterance)
    }

    private func _announceCriticalAlerts(_ alerts: [MaintenanceAlert]) async {
        guard !alerts.isEmpty else { return }

        // Rate-limiting: avoid repeating announcements too often
        if let last = lastAnnouncementDate, Date().timeIntervalSince(last) < announcementCooldown {
            #if DEBUG
            print("SpeechAlertManager: Skipping announcement due to cooldown.")
            #endif
            return
        }
        lastAnnouncementDate = Date()

        // Filter out alerts we've already announced in this session
        let newAlerts = alerts.filter { !self.lastAnnouncedIDs.contains($0.id) }

        guard !newAlerts.isEmpty else {
            #if DEBUG
            print("SpeechAlertManager: No new critical alerts to announce.")
            #endif
            return
        }

        // Limit how many alerts we combine in one utterance
        let limitedAlerts = Array(newAlerts.prefix(maxCombinedAlerts))

        // Build concise combined message
        let combinedMessage = buildCombinedMessage(from: limitedAlerts)

        // Ensure we have a voice
        if preferredVoice == nil {
            await loadPreferredVoiceSafely()
        }

        let utterance = makeUtterance(from: combinedMessage)

        // Speak (or schedule if busy)
        if !synthesizer.isSpeaking {
            speakUtterance(utterance)
            limitedAlerts.forEach { lastAnnouncedIDs.insert($0.id) }
            #if DEBUG
            print("SpeechAlertManager: Announced \(limitedAlerts.count) critical alert(s).")
            #endif
        } else {
            // If already speaking, schedule a short retry rather than overlapping
            #if DEBUG
            print("SpeechAlertManager: Synthesizer busy; scheduling a delayed announcement.")
            #endif
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                Task { @MainActor in
                    if !self.synthesizer.isSpeaking {
                        self.speakUtterance(utterance)
                        limitedAlerts.forEach { self.lastAnnouncedIDs.insert($0.id) }
                        #if DEBUG
                        print("SpeechAlertManager: Announced (delayed) \(limitedAlerts.count) critical alert(s).")
                        #endif
                    } else {
                        #if DEBUG
                        print("SpeechAlertManager: Still busy; skipping delayed announcement.")
                        #endif
                    }
                }
            }
        }
    }

    // MARK: - Helpers

    private func makeUtterance(from text: String) -> AVSpeechUtterance {
        let utterance = AVSpeechUtterance(string: text)
        // Keep speech rate moderate — avoid too fast in some voices
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.95
        utterance.preUtteranceDelay = 0.08
        utterance.postUtteranceDelay = 0.12
        utterance.voice = preferredVoice ?? AVSpeechSynthesisVoice(language: "en-US")
        return utterance
    }

    private func speakUtterance(_ utterance: AVSpeechUtterance) {
        // Configure audio session each time we speak in case audio system state changed
        configureAudioSession()
        synthesizer.speak(utterance)
    }

    private func buildCombinedMessage(from alerts: [MaintenanceAlert]) -> String {
        // Keep it short — component + one-line message, join with ". "
        var parts: [String] = []
        for alert in alerts {
            let component = alert.component.trimmingCharacters(in: .whitespacesAndNewlines)
            // Keep message short
            let shortMsg = alert.message.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
            parts.append("\(component): \(shortMsg)")
        }
        let joined = parts.joined(separator: ". ")
        // Limit total length to avoid huge utterances
        if joined.count > 300 {
            let truncated = String(joined.prefix(300))
            return "Critical alerts. \(truncated)... Please check the vehicle."
        }
        return "Critical alerts. \(joined). Please check the vehicle."
    }

    // Configure AVAudioSession for spoken audio
    private func configureAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: [])
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            // Non-fatal — log for debugging
            #if DEBUG
            print("SpeechAlertManager: AVAudioSession config error:", error)
            #endif
        }
    }

    // Load preferred voice with defensive fallbacks
    private func loadPreferredVoiceSafely() async {
        // This is defensive — AVSpeechSynthesisVoice.speechVoices() can sometimes throw internal decode errors in Simulator.
        do {
            // In practice speechVoices() doesn't throw, but guarding with catch in case frameworks misbehave.
            let voices = AVSpeechSynthesisVoice.speechVoices()
            // Prefer system English voices (en-*, en_US, etc.)
            if let enVoice = voices.first(where: { $0.language.starts(with: "en") }) {
                preferredVoice = enVoice
            } else {
                preferredVoice = AVSpeechSynthesisVoice(language: "en-US")
            }
        } catch {
            // Fallback if fetching voices fails (e.g., Simulator specific issue)
            #if DEBUG
            print("SpeechAlertManager: Error fetching voices, using fallback voice:", error)
            #endif
            preferredVoice = AVSpeechSynthesisVoice(language: "en-US")
        }
    }
}

// MARK: - AVSpeechSynthesizerDelegate
extension SpeechAlertManager: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        #if DEBUG
        print("SpeechAlertManager: started utterance.")
        #endif
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        #if DEBUG
        print("SpeechAlertManager: finished utterance.")
        #endif
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        #if DEBUG
        print("SpeechAlertManager: cancelled utterance.")
        #endif
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didPause utterance: AVSpeechUtterance) {
        #if DEBUG
        print("SpeechAlertManager: paused utterance.")
        #endif
    }
}
