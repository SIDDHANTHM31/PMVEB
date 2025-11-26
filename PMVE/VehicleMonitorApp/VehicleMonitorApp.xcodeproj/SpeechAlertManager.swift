import AVFoundation
import Foundation

/// A speech alert manager for the iOS app UI that announces critical maintenance alerts.
/// 
/// This manager is intended for non-CarPlay use only and does not interact with CarPlay at all.
/// It provides spoken announcements of critical alerts with deduplication and cooldown to avoid repetition.
/// 
/// Usage:
/// ```
/// SpeechAlertManager.shared.announceCriticalAlerts(alerts)
/// ```
///
/// The class ensures announcements are performed on the main thread.
@MainActor
public final class SpeechAlertManager {
    /// Shared singleton instance
    public static let shared = SpeechAlertManager()

    private let synthesizer = AVSpeechSynthesizer()
    private var lastSpokenAlerts: [String: Date] = [:]
    private let cooldownInterval: TimeInterval = 60

    private init() {}

    /// Announces the given critical maintenance alerts via speech synthesis.
    ///
    /// Only alerts with `priority` equal to "critical" (case-insensitive) are spoken.
    /// Alerts with the same component and message will not be repeated within 60 seconds.
    ///
    /// - Parameter alerts: The array of `MaintenanceAlert` to announce.
    public func announceCriticalAlerts(_ alerts: [MaintenanceAlert]) {
        let now = Date()
        let criticalAlerts = alerts.filter { $0.priority.lowercased() == "critical" }

        for alert in criticalAlerts {
            let key = "\(alert.component)|\(alert.message)"
            if let lastSpoken = lastSpokenAlerts[key], now.timeIntervalSince(lastSpoken) < cooldownInterval {
                continue
            }

            let utteranceString = "Critical alert for \(alert.component). \(alert.message). Recommended action: \(alert.recommendedAction)."
            let utterance = AVSpeechUtterance(string: utteranceString)
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
            utterance.rate = AVSpeechUtteranceDefaultSpeechRate
            utterance.prefersAssistiveTechnologySettings = true

            if synthesizer.isSpeaking {
                synthesizer.stopSpeaking(at: .immediate)
            }

            synthesizer.speak(utterance)
            lastSpokenAlerts[key] = now
        }
    }

    /// Stops any ongoing speech synthesis immediately.
    public func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}
