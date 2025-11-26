import SwiftUI

struct ContentView: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    @EnvironmentObject var networkManager: NetworkManager
    @State private var selectedTab = 0
    @State private var showingAlert = false
    @State private var alertMessage = ""
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Dashboard Tab
            DashboardView()
                .tabItem {
                    Image(systemName: "gauge")
                    Text("Dashboard")
                }
                .tag(0)
            
            // Engine Tab
            EngineStatusView()
                .tabItem {
                    Image(systemName: "engine.combustion")
                    Text("Engine")
                }
                .tag(1)
            
            // Battery Tab
            BatteryStatusView()
                .tabItem {
                    Image(systemName: "battery.100")
                    Text("Battery")
                }
                .tag(2)
            
            // Alerts Tab
            MaintenanceAlertsView()
                .tabItem {
                    Image(systemName: "exclamationmark.triangle")
                    Text("Alerts")
                }
                .badge(vehicleData.criticalAlertsCount > 0 ? String(vehicleData.criticalAlertsCount) : nil)
                .tag(3)
        }
        // Use mapped color from vehicleData.vehicleStatusColor (string -> system Color)
        .accentColor(vehicleData.vehicleStatusColor)
        .onAppear {
            setupVehicleDataReference()
        }
        .onChange(of: vehicleData.hasActiveCriticalAlerts) { hasCriticalAlerts in
            if hasCriticalAlerts {
                selectedTab = 3 // Switch to alerts tab
            }
        }
    }
    
    private func setupVehicleDataReference() {
        VehicleDataModel.shared = vehicleData
    }
}

// MARK: - Dashboard View
struct DashboardView: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    @EnvironmentObject var networkManager: NetworkManager
    @AppStorage("voiceAlertsEnabled") private var voiceAlertsEnabled: Bool = true
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Connection Status
                    ConnectionStatusCard()
                    
                    // Overall Vehicle Health
                    VehicleHealthCard()
                    
                    // Quick Status Cards
                    HStack(spacing: 15) {
                        QuickStatusCard(
                            title: "Engine",
                            value: vehicleData.engineData?.condition ?? "Unknown",
                            health: vehicleData.engineData?.health ?? 0,
                            color: (vehicleData.engineData?.statusColor ?? .gray),
                            systemImage: "engine.combustion"
                        )
                        
                        QuickStatusCard(
                            title: "Battery",
                            value: vehicleData.batteryData?.condition ?? "Unknown",
                            health: vehicleData.batteryData?.health ?? 0,
                            color: (vehicleData.batteryData?.statusColor ?? .gray),
                            systemImage: "battery.100"
                        )
                    }
                    
                    // Critical Alerts Section
                    if vehicleData.hasActiveCriticalAlerts {
                        CriticalAlertsCard()
                    }
                    
                    // Voice Alerts Setting
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle(isOn: $voiceAlertsEnabled) {
                            HStack(spacing: 8) {
                                Image(systemName: "speaker.wave.2.fill")
                                    .foregroundColor(.blue)
                                Text("Voice Alerts")
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                            }
                        }
                        .tint(.blue)

                        Text("When enabled, critical alerts will be announced automatically.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    
                    // Recent Data Section
                    RecentDataCard()
                    TestVoiceAlertButton()
                }
                .padding()
            }
            .navigationTitle("Vehicle Monitor")
            .refreshable {
                networkManager.fetchVehicleData()
            }
        }
    }
}

// MARK: - Connection Status Card
struct ConnectionStatusCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        HStack {
            Circle()
                .fill(vehicleData.isConnected ? Color.green : Color.red)
                .frame(width: 12, height: 12)
            
            Text(vehicleData.connectionStatus)
                .font(.headline)
            
            Spacer()
            
            if let lastUpdate = vehicleData.lastUpdateTime {
                Text("Updated \(lastUpdate, formatter: timeFormatter)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
    
    private var timeFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.timeStyle = .medium
        return formatter
    }
}

// MARK: - Vehicle Health Card (unchanged, uses system colors)
struct VehicleHealthCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(spacing: 15) {
            Text("Overall Vehicle Health")
                .font(.title2)
                .fontWeight(.semibold)
            
            ZStack {
                Circle()
                    .stroke(Color(.systemGray4), lineWidth: 10)
                    .frame(width: 150, height: 150)
                
                Circle()
                    .trim(from: 0, to: CGFloat(vehicleData.overallVehicleHealth / 100))
                    .stroke(vehicleData.vehicleStatusColor, lineWidth: 10)
                    .rotationEffect(.degrees(-90))
                    .frame(width: 150, height: 150)
                
                VStack {
                    Text("\(Int(vehicleData.overallVehicleHealth))%")
                        .font(.title)
                        .fontWeight(.bold)
                    Text(vehicleData.vehicleStatusSummary)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 5)
    }
}

// MARK: - Quick Status Card
struct QuickStatusCard: View {
    let title: String
    let value: String
    let health: Double
    let color: Color
    let systemImage: String
    
    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.title2)
                .foregroundColor(color)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(value)
                .font(.headline)
                .fontWeight(.semibold)
            
            Text("\(Int(health))%")
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(color.opacity(0.2))
                .foregroundColor(color)
                .cornerRadius(8)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

// MARK: - Critical Alerts Card (uses system red safely)
struct CriticalAlertsCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.red)
                
                Text("Critical Alerts")
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Spacer()
                
                Text("\(vehicleData.criticalAlertsCount)")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.red.opacity(0.2))
                    .foregroundColor(.red)
                    .cornerRadius(8)
            }
            
            ForEach(vehicleData.maintenanceAlerts.prefix(3)) { alert in
                if alert.priority.lowercased() == "critical" {
                    HStack {
                        VStack(alignment: .leading) {
                            Text(alert.component)
                                .font(.subheadline)
                                .fontWeight(.medium)
                            Text(alert.message)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                        
                        Text("\(alert.daysRemaining)d")
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(.red)
                    }
                    .padding(.vertical, 5)
                }
            }
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(10)
    }
}

// Rest of the ContentView file (RecentDataCard, TestVoiceAlertButton, SensorDataRow, MaintenanceAlertsView, MaintenanceAlertRow, Previews)
// — keep existing implementations from your file; they already use system colors for most controls.


// MARK: - Recent Data Card
struct RecentDataCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Recent Sensor Data")
                .font(.headline)
                .fontWeight(.semibold)
            
            if let engine = vehicleData.engineData {
                VStack(spacing: 8) {
                    SensorDataRow(label: "Engine RPM", value: "\(Int(engine.rpm))", unit: "RPM")
                    SensorDataRow(label: "Oil Pressure", value: String(format: "%.1f", engine.oilPressure), unit: "PSI")
                    SensorDataRow(label: "Coolant Temp", value: String(format: "%.1f", engine.coolantTemperature), unit: "°C")
                }
            }
            
            if let battery = vehicleData.batteryData {
                VStack(spacing: 8) {
                    SensorDataRow(label: "Battery Voltage", value: String(format: "%.2f", battery.voltage), unit: "V")
                    SensorDataRow(label: "Battery Temp", value: String(format: "%.1f", battery.temperature), unit: "°C")
                    SensorDataRow(label: "Capacity", value: String(format: "%.1f", battery.capacity), unit: "%")
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

struct TestVoiceAlertButton: View {
    var body: some View {
        Button {
            // Create a sample critical alert to speak
            let sample = MaintenanceAlert(
                component: "Engine Oil",
                priority: "Critical",
                message: "Oil pressure critically low",
                recommendedAction: "Stop vehicle safely and check oil level",
                daysRemaining: 0,
                timestamp: Date()
            )
            SpeechAlertManager.shared.announceCriticalAlerts([sample])
        } label: {
            HStack {
                Image(systemName: "speaker.wave.2.fill")
                Text("Test Voice Alert")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.blue.opacity(0.15))
            .foregroundColor(.blue)
            .cornerRadius(10)
        }
        .buttonStyle(.plain)
        .padding(.top, 4)
    }
}

// MARK: - Sensor Data Row
struct SensorDataRow: View {
    let label: String
    let value: String
    let unit: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
            
            Spacer()
            
            HStack(spacing: 4) {
                Text(value)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
}

// MARK: - Maintenance Alerts View
struct MaintenanceAlertsView: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        NavigationView {
            List {
                ForEach(vehicleData.maintenanceAlerts) { alert in
                    MaintenanceAlertRow(alert: alert)
                }
            }
            .navigationTitle("Maintenance Alerts")
            .refreshable {
                NetworkManager.shared.fetchMaintenanceAlerts()
            }
        }
    }
}

// MARK: - Maintenance Alert Row
struct MaintenanceAlertRow: View {
    let alert: MaintenanceAlert
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text(alert.component)
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Spacer()
                    
                    Text(alert.priority.uppercased())
                        .font(.caption)
                        .fontWeight(.bold)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(alert.priorityColor.opacity(0.2))
                        .foregroundColor(alert.priorityColor)
                        .cornerRadius(8)
                }
                
                Text(alert.message)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Text(alert.recommendedAction)
                    .font(.caption)
                    .foregroundColor(.blue)
                
                Text("\(alert.daysRemaining) days remaining")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
        }
        .padding(.vertical, 5)
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(NetworkManager.shared)
            .environmentObject(VehicleDataModel.sampleData())
    }
}

