import Foundation
import Starscream
import Combine

// MARK: - Data Models (Unified - No Duplicates)
struct VehicleAPIResponse: Codable {
    let timestamp: Int
    let vehicleId: String
    let engine: EngineInfo
    let battery: BatteryInfo
    let overall: OverallStatus
    let recommendations: RecommendationsData
    let metadata: MetadataInfo

    enum CodingKeys: String, CodingKey {
        case timestamp
        case vehicleId = "vehicle_id"
        case engine, battery, overall, recommendations, metadata
    }
}

struct EngineInfo: Codable {
    let condition: String
    let conditionCode: Int
    let healthScore: Double
    let sensorData: EngineSensorInfo
    let partsHealth: [String: Double]

    enum CodingKeys: String, CodingKey {
        case condition
        case conditionCode = "condition_code"
        case healthScore = "health_score"
        case sensorData = "sensor_data"
        case partsHealth = "parts_health"
    }
}

struct EngineSensorInfo: Codable {
    let rpm: Double
    let lubOilPressure: Double
    let fuelPressure: Double
    let coolantPressure: Double
    let lubOilTemp: Double
    let coolantTemp: Double
    let viscosity: Double

    enum CodingKeys: String, CodingKey {
        case rpm
        case lubOilPressure = "lub_oil_pressure"
        case fuelPressure = "fuel_pressure"
        case coolantPressure = "coolant_pressure"
        case lubOilTemp = "lub_oil_temp"
        case coolantTemp = "coolant_temp"
        case viscosity
    }
}

struct BatteryInfo: Codable {
    let condition: String
    let healthScore: Double
    let priorityLevel: String
    let sensorData: BatterySensorInfo
    let remainingMonths: Double
    let remainingDays: Int
    let safetyWarning: String
    let isRealData: Bool

    enum CodingKeys: String, CodingKey {
        case condition
        case healthScore = "health_score"
        case priorityLevel = "priority_level"
        case sensorData = "sensor_data"
        case remainingMonths = "remaining_months"
        case remainingDays = "remaining_days"
        case safetyWarning = "safety_warning"
        case isRealData = "is_real_data"
    }
}

struct BatterySensorInfo: Codable {
    let voltage: Double
    let current: Double
    let temperature: Double
    let capacity: Double
    let internalResistance: Double
    let chargeCycles: Double

    enum CodingKeys: String, CodingKey {
        case voltage, current, temperature, capacity
        case internalResistance = "internal_resistance"
        case chargeCycles = "charge_cycles"
    }
}

struct OverallStatus: Codable {
    let vehicleHealth: Double
    let engineHealth: Double
    let batteryHealth: Double
    let status: String

    enum CodingKeys: String, CodingKey {
        case vehicleHealth = "vehicle_health"
        case engineHealth = "engine_health"
        case batteryHealth = "battery_health"
        case status
    }
}

struct RecommendationsData: Codable {
    let engine: [EngineRecommendation]
    let battery: [BatteryRecommendation]
}

struct EngineRecommendation: Codable, Identifiable {
    var id: String { part }
    let part: String
    let currentHealth: String
    let healthScore: Double
    let priority: String
    let priorityLevel: String
    let recommendedReplacementDate: String
    let remainingDays: Int
    let remainingKilometers: Double
    let remainingKmFormatted: String

    enum CodingKeys: String, CodingKey {
        case part
        case currentHealth = "current_health"
        case healthScore = "health_score"
        case priority
        case priorityLevel = "priority_level"
        case recommendedReplacementDate = "recommended_replacement_date"
        case remainingDays = "remaining_days"
        case remainingKilometers = "remaining_kilometers"
        case remainingKmFormatted = "remaining_km_formatted"
    }
}

struct BatteryRecommendation: Codable, Identifiable {
    var id: String { component }
    let component: String
    let currentHealth: String
    let healthScore: Double
    let condition: String
    let priority: String
    let priorityLevel: String
    let recommendedReplacementDate: String
    let remainingDays: Int
    let remainingMonths: Double
    let voltage: Double
    let temperature: Double
    let internalResistance: Double
    let safetyWarning: String

    enum CodingKeys: String, CodingKey {
        case component
        case currentHealth = "current_health"
        case healthScore = "health_score"
        case condition, priority
        case priorityLevel = "priority_level"
        case recommendedReplacementDate = "recommended_replacement_date"
        case remainingDays = "remaining_days"
        case remainingMonths = "remaining_months"
        case voltage, temperature
        case internalResistance = "internal_resistance"
        case safetyWarning = "safety_warning"
    }
}

struct MetadataInfo: Codable {
    let engineSensorsCount: Int
    let batterySensorsCount: Int
    let historyPointsEngine: Int
    let historyPointsBattery: Int
    let batteryDataSource: String

    enum CodingKeys: String, CodingKey {
        case engineSensorsCount = "engine_sensors_count"
        case batterySensorsCount = "battery_sensors_count"
        case historyPointsEngine = "history_points_engine"
        case historyPointsBattery = "history_points_battery"
        case batteryDataSource = "battery_data_source"
    }
}

// MARK: - Alert Model for Speech
struct VehicleMaintenanceAlert {
    let component: String
    let priority: String
    let message: String
    let recommendedAction: String
}

// Local MaintenanceAlert (used by SpeechAlertManager)
struct MaintenanceAlert {
    let component: String
    let priority: String
    let message: String
    let recommendedAction: String
    let daysRemaining: Int
}

// Simple SpeechAlertManager stub — replace with your real implementation
final class SpeechAlertManager {
    static let shared = SpeechAlertManager()
    private init() {}

    func announceCriticalAlerts(_ alerts: [MaintenanceAlert]) {
        // Replace with actual TTS / notification implementation
        for alert in alerts {
            print("ALERT: [\(alert.priority.uppercased())] \(alert.component) — \(alert.message) -> \(alert.recommendedAction)")
        }
    }
}

// MARK: - WebSocket Service
@MainActor
class VehicleDataService: ObservableObject, WebSocketDelegate {
    static let shared = VehicleDataService()

    // CHANGE THIS TO YOUR COMPUTER'S IP ADDRESS IF USING A PHYSICAL DEVICE
    private let serverURL = "http://localhost:5000"

    @Published var vehicleData: VehicleAPIResponse?
    @Published var isConnected: Bool = false
    @Published var connectionError: String?
    @Published var lastUpdateTime: Date?

    private var socket: WebSocket?
    private var pollingTimer: Timer?
    private var isPolling = false

    private init() {}

    // MARK: - HTTP Polling (Recommended - More Reliable)
    func startPolling() {
        guard !isPolling else { return }
        isPolling = true

        pollingTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.fetchVehicleData()
            }
        }

        Task { @MainActor in
            await fetchVehicleData()
        }

        print("📱 Started polling for vehicle data")
    }

    func stopPolling() {
        pollingTimer?.invalidate()
        pollingTimer = nil
        isPolling = false
        print("📱 Stopped polling")
    }

    private func fetchVehicleData() async {
        guard let url = URL(string: "\(serverURL)/api/vehicle-status") else {
            connectionError = "Invalid URL"
            isConnected = false
            return
        }

        do {
            let (data, response) = try await URLSession.shared.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                connectionError = "Server error"
                isConnected = false
                return
            }

            if data.isEmpty || String(data: data, encoding: .utf8) == "{}" {
                isConnected = true
                return
            }

            let decoder = JSONDecoder()
            let vehicleData = try decoder.decode(VehicleAPIResponse.self, from: data)

            self.vehicleData = vehicleData
            self.isConnected = true
            self.connectionError = nil
            self.lastUpdateTime = Date()

            checkForCriticalAlerts(vehicleData)
        } catch {
            print("❌ Error fetching data: \(error)")
            connectionError = error.localizedDescription
            isConnected = false
        }
    }

    // MARK: - WebSocket Delegate
    nonisolated func didReceive(event: WebSocketEvent, client: WebSocketClient) {
        Task { @MainActor in
            switch event {
            case .connected(_):
                isConnected = true
                connectionError = nil
                print("📱 WebSocket connected")

            case .disconnected(let reason, _):
                isConnected = false
                connectionError = reason
                print("📱 WebSocket disconnected: \(reason)")

            case .text(let text):
                handleWebSocketMessage(text)

            case .binary(_):
                // ignoring binary payloads for now
                break

            case .error(let error):
                isConnected = false
                connectionError = error?.localizedDescription
                print("❌ WebSocket error: \(String(describing: error))")

            case .cancelled:
                isConnected = false
                print("📱 WebSocket cancelled")

            default:
                break
            }
        }
    }

    private func handleWebSocketMessage(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }

        do {
            let decoder = JSONDecoder()
            let vehicleData = try decoder.decode(VehicleAPIResponse.self, from: data)

            self.vehicleData = vehicleData
            self.lastUpdateTime = Date()

            checkForCriticalAlerts(vehicleData)
        } catch {
            print("❌ Error decoding WebSocket message: \(error)")
        }
    }

    // MARK: - Critical Alerts
    private func checkForCriticalAlerts(_ data: VehicleAPIResponse) {
        var alerts: [VehicleMaintenanceAlert] = []

        if data.engine.condition.lowercased() == "bad" {
            alerts.append(VehicleMaintenanceAlert(
                component: "Engine",
                priority: "critical",
                message: "Engine condition is critical",
                recommendedAction: "Immediate inspection required"
            ))
        }

        if data.battery.priorityLevel.uppercased() == "CRITICAL" {
            let message = data.battery.safetyWarning.isEmpty ? "Battery needs immediate replacement" : data.battery.safetyWarning
            alerts.append(VehicleMaintenanceAlert(
                component: "Battery",
                priority: "critical",
                message: message,
                recommendedAction: "Replace battery immediately"
            ))
        }

        for recommendation in data.recommendations.engine {
            if recommendation.priorityLevel.uppercased() == "CRITICAL" {
                alerts.append(VehicleMaintenanceAlert(
                    component: recommendation.part,
                    priority: "critical",
                    message: "Health at \(recommendation.currentHealth)",
                    recommendedAction: recommendation.priority
                ))
            }
        }

        if !alerts.isEmpty {
            let maintenanceAlerts = alerts.map { alert in
                MaintenanceAlert(
                    component: alert.component,
                    priority: alert.priority,
                    message: alert.message,
                    recommendedAction: alert.recommendedAction,
                    daysRemaining: 0
                )
            }
            SpeechAlertManager.shared.announceCriticalAlerts(maintenanceAlerts)
        }
    }
}
