import Foundation
import Combine
import SwiftUI

class NetworkManager: ObservableObject {
    static let shared = NetworkManager()
    
    @AppStorage("voiceAlertsEnabled") private var voiceAlertsEnabled: Bool = true

    @Published var isConnected: Bool = false
    @Published var connectionError: String?
    
    private var cancellables = Set<AnyCancellable>()
    private var monitoringTimer: Timer?
    private var isRequestInFlight = false
    
    // Configure your Flask backend URL here (endpoints: /predict)
    // For Simulator, 127.0.0.1 points to your Mac. For a physical device, use your Mac's LAN IP (e.g., http://192.168.x.x:5002)
    private let baseURL = "http://127.0.0.1:5002"
    
    private init() {}
    
    // MARK: - Public Methods
    func startMonitoring() {
        stopMonitoring()
        
        // Start periodic data fetching every 3 seconds
        monitoringTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { _ in
            self.fetchVehicleData()
        }
        
        // Fetch initial data
        fetchVehicleData()
    }
    
    func stopMonitoring() {
        monitoringTimer?.invalidate()
        monitoringTimer = nil
    }
    
    func fetchVehicleData() {
        guard !isRequestInFlight else { return }
        isRequestInFlight = true
        
        guard let url = URL(string: "\(baseURL)/predict") else {
            updateConnectionStatus(false, error: "Invalid URL")
            isRequestInFlight = false
            return
        }

        // Prepare payload using last known values or sensible defaults
        let last = VehicleDataModel.shared
        let payload: [String: Any] = [
            "engineRpm": last?.engineData?.rpm ?? 1500,
            "lubOilPressure": last?.engineData?.oilPressure ?? 50,
            "fuelPressure": last?.engineData?.fuelPressure ?? 3.5,
            "coolantPressure": last?.engineData?.coolantPressure ?? 2.0,
            "lubOilTemp": last?.engineData?.oilTemperature ?? 80,
            "coolantTemp": last?.engineData?.coolantTemperature ?? 85,
            "viscosity": 10.0 // backend will compute if lubOilTemp present; send default
        ]

        guard let body = try? JSONSerialization.data(withJSONObject: payload, options: []) else {
            updateConnectionStatus(false, error: "Failed to encode request body")
            isRequestInFlight = false
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = body
        request.timeoutInterval = 8.0
        request.cachePolicy = .reloadIgnoringLocalCacheData

        URLSession.shared.dataTaskPublisher(for: request)
            .tryMap { output -> Data in
                guard let response = output.response as? HTTPURLResponse else {
                    throw URLError(.badServerResponse)
                }
                guard (200..<300).contains(response.statusCode) else {
                    let bodyString = String(data: output.data, encoding: .utf8) ?? "<no body>"
                    throw NSError(
                        domain: "NetworkManager",
                        code: response.statusCode,
                        userInfo: [NSLocalizedDescriptionKey: "HTTP \(response.statusCode): \(bodyString)"]
                    )
                }
                return output.data
            }
            .decode(type: PredictResponse.self, decoder: JSONDecoder.vehicleDecoder)
            .receive(on: DispatchQueue.main)
            .sink(
                receiveCompletion: { [weak self] completion in
                    guard let self = self else { return }
                    self.isRequestInFlight = false
                    switch completion {
                    case .failure(let error):
                        self.updateConnectionStatus(false, error: error.localizedDescription)
                        #if DEBUG
                        print("fetchVehicleData error:", error)
                        #endif
                    case .finished:
                        break
                    }
                },
                receiveValue: { [weak self] response in
                    guard let self = self else { return }
                    
                    // ===== DEBUG: show decoded response info =====
                    #if DEBUG
                    print("✅ NETWORK: decoded PredictResponse -> overall_status:", response.overall_status)
                    print("✅ NETWORK: current_readings keys:", Array(response.current_readings.keys))
                    if let er = response.current_readings["engineRpm"] {
                        print("✅ NETWORK: engineRpm decoded as:", er)
                    } else {
                        // show alternatives if engineRpm missing
                        print("✅ NETWORK: engineRpm missing; keys available:", Array(response.current_readings.keys))
                    }
                    print("✅ NETWORK: recommendations count:", response.recommendations.count)
                    #endif
                    // ===============================================
                    
                    self.updateConnectionStatus(true)
                    self.processPredictResponse(response)
                    self.isRequestInFlight = false
                }
            )
            .store(in: &cancellables)
    }
    
    func fetchMaintenanceAlerts() {
        // Backend no longer provides a dedicated alerts endpoint; alerts are part of /predict response.
        // Trigger a fetch to update recommendations-derived alerts.
        fetchVehicleData()
    }
    
    // MARK: - Private Methods
    private func updateConnectionStatus(_ connected: Bool, error: String? = nil) {
        DispatchQueue.main.async {
            self.isConnected = connected
            self.connectionError = error
            
            if let vehicleData = VehicleDataModel.shared {
                vehicleData.updateConnectionStatus(connected, status: error ?? (connected ? "Connected" : "Disconnected"))
            }
        }
    }
    
    private func processPredictResponse(_ response: PredictResponse) {
        #if DEBUG
        print(">>> processPredictResponse called. overall_status:", response.overall_status)
        print(">>> current_readings keys:", Array(response.current_readings.keys))
        #endif
        
        guard let vehicleData = VehicleDataModel.shared else {
            #if DEBUG
            print(">>> processPredictResponse: VehicleDataModel.shared is nil — aborting updates")
            #endif
            return
        }

        // Derive a health score from parts_health average
        let healthValues = Array(response.parts_health.values)
        let avgHealth = healthValues.isEmpty ? 0.0 : (healthValues.reduce(0, +) / Double(healthValues.count))

        // Map current_readings to engine (decoder is robust; still try common alternate keys)
        let readings = response.current_readings
        
        // Try multiple keys for resilience (backends sometimes change naming)
        func valueForKeys(_ keys: [String], default defaultVal: Double) -> Double {
            for k in keys {
                if let v = readings[k]?.value { return v }
            }
            return defaultVal
        }

        let rpm = valueForKeys(["engineRpm", "engine_rpm", "rpm"], default: 1500)
        let oilP = valueForKeys(["lubOilPressure", "oilPressure", "oil_pressure"], default: 50)
        let fuelP = valueForKeys(["fuelPressure", "fuel_pressure", "fuelP"], default: 3.5)
        let coolP = valueForKeys(["coolantPressure", "coolant_pressure", "coolantP"], default: 2.0)
        let oilT = valueForKeys(["lubOilTemp", "oilTemp", "lubOil_Temp"], default: 80)
        let coolT = valueForKeys(["coolantTemp", "coolant_temp"], default: 85)

        let engine = EngineData(
            rpm: rpm,
            oilPressure: oilP,
            fuelPressure: fuelP,
            coolantPressure: coolP,
            oilTemperature: oilT,
            coolantTemperature: coolT,
            condition: response.overall_status,
            health: avgHealth,
            timestamp: Date()
        )
        vehicleData.updateEngineData(engine)

        // Synthesize battery data from status
        let voltage: Double
        switch response.overall_status.lowercased() {
        case "good": voltage = 12.6
        case "bad": voltage = 11.8
        default: voltage = 12.2
        }

        let battery = BatteryData(
            voltage: voltage,
            current: 0.0,
            temperature: coolT,
            capacity: avgHealth,
            internalResistance: 0.05,
            chargeCycles: 300,
            condition: response.overall_status,
            health: avgHealth,
            timestamp: Date()
        )
        vehicleData.updateBatteryData(battery)

        // Convert recommendations to MaintenanceAlert list
        let alerts: [MaintenanceAlert] = response.recommendations.map { rec in
            let days: Int = {
                let comps = rec.approx_time_left.split(separator: " ")
                if let first = comps.first, let n = Int(first) { return max(1, n) }
                return 7
            }()
            return MaintenanceAlert(
                component: rec.part,
                priority: rec.priority_level.capitalized,
                message: "\(rec.priority): \(rec.remaining_km_formatted), \(rec.approx_time_left)",
                recommendedAction: "Inspect/Service soon",
                daysRemaining: days,
                timestamp: Date()
            )
        }
        vehicleData.updateMaintenanceAlerts(alerts)

        // Automatic voice announcements for critical alerts (respect user setting)
        if voiceAlertsEnabled {
            let criticalAlerts = alerts.filter { $0.priority.lowercased() == "critical" }
            if !criticalAlerts.isEmpty {
                Task { @MainActor in
                    SpeechAlertManager.shared.announceCriticalAlerts(criticalAlerts)
                }
            }
        }
    }
}

// MARK: - Predict Response Models
struct PredictResponse: Codable {
    let overall_status: String
    let parts_health: [String: Double]
    let recommendations: [Recommendation]
    let current_readings: [String: DoubleCodable]
}

struct Recommendation: Codable {
    let part: String
    let current_health: String
    let health_score: Double
    let priority: String
    let priority_level: String
    let recommended_replacement_date: String
    let remaining_kilometers: Double
    let remaining_km_formatted: String
    let approx_time_left: String
}

// Robust DoubleCodable — accepts {"value": number} OR raw number (Double/Int/String)
struct DoubleCodable: Codable, CustomDebugStringConvertible {
    let value: Double

    var debugDescription: String { "DoubleCodable(value: \(value))" }

    init(_ value: Double) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        // First try to decode a raw single value (Double / Int / String)
        let container = try decoder.singleValueContainer()
        if let d = try? container.decode(Double.self) {
            self.value = d
            return
        }
        if let i = try? container.decode(Int.self) {
            self.value = Double(i)
            return
        }
        if let s = try? container.decode(String.self), let d = Double(s) {
            self.value = d
            return
        }

        // If raw decode failed, try keyed container with "value" key
        let keyed = try decoder.container(keyedBy: CodingKeys.self)
        if let v = try? keyed.decode(Double.self, forKey: .value) {
            self.value = v
            return
        }
        if let vi = try? keyed.decode(Int.self, forKey: .value) {
            self.value = Double(vi)
            return
        }
        if let vs = try? keyed.decode(String.self, forKey: .value), let d = Double(vs) {
            self.value = d
            return
        }

        // Fallback default
        self.value = 0.0
    }

    func encode(to encoder: Encoder) throws {
        // Encode as object with value (keeps parity with server expectation)
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(value, forKey: .value)
    }

    private enum CodingKeys: String, CodingKey {
        case value
    }
}

// MARK: - Extensions
extension JSONDecoder {
    static let vehicleDecoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
}

// Add to VehicleDataModel
extension VehicleDataModel {
    static var shared: VehicleDataModel?
}
