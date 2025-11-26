import Foundation
import AVFoundation

@MainActor
public final class SpeechAlertManager {
    public static let shared = SpeechAlertManager()

    private let synthesizer = AVSpeechSynthesizer()
    private var lastSpokenAlerts: [String: Date] = [:]
    private let cooldownInterval: TimeInterval = 60

    private init() {}

    public func announceCriticalAlerts(_ alerts: [MaintenanceAlert]) {
        let now = Date()
        // Filter to critical alerts defensively
        let criticalAlerts = alerts.filter { $0.priority.lowercased() == "critical" }
        guard !criticalAlerts.isEmpty else { return }

        for alert in criticalAlerts {
            let key = "\(alert.component)|\(alert.message)"
            if let lastSpoken = lastSpokenAlerts[key], now.timeIntervalSince(lastSpoken) < cooldownInterval {
                continue
            }

            let utteranceText = "Critical alert for \(alert.component). \(alert.message). Recommended action: \(alert.recommendedAction)."
            let utterance = AVSpeechUtterance(string: utteranceText)
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

    public func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}
