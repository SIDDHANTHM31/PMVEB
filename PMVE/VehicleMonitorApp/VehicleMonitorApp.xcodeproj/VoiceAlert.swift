import Foundation
import AVFoundation

/// A simple text-to-speech voice alert service.
final class VoiceAlert: ObservableObject {
    private let synthesizer = AVSpeechSynthesizer()

    /// Speaks the provided text using system TTS.
    /// - Parameters:
    ///   - text: The message to speak.
    ///   - language: BCP-47 language code (default: en-US).
    ///   - rate: Speech rate (0.0...1.0 roughly maps to slow...fast). Defaults to AVSpeechUtteranceDefaultSpeechRate.
    func speak(_ text: String, language: String = "en-US", rate: Float = AVSpeechUtteranceDefaultSpeechRate) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: language)
        utterance.rate = rate
        // Stop any ongoing speech immediately to avoid overlap
        synthesizer.stopSpeaking(at: .immediate)
        synthesizer.speak(utterance)
    }

    /// Stop any ongoing speech.
    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}
