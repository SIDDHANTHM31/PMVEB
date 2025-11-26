import SwiftUI

struct EngineStatusView: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    @EnvironmentObject var networkManager: NetworkManager
    @State private var showingDetails = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Engine Health Overview
                    EngineHealthCard()
                    
                    // Real-time Engine Metrics
                    EngineMetricsGrid()
                    
                    // Engine Performance Chart (simplified)
                    EnginePerformanceCard()
                    
                    // Engine Alerts
                    EngineAlertsCard()
                }
                .padding()
            }
            .navigationTitle("Engine Status")
            .navigationBarTitleDisplayMode(.large)
            .refreshable {
                networkManager.fetchVehicleData()
            }
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Details") {
                        showingDetails = true
                    }
                }
            }
            .sheet(isPresented: $showingDetails) {
                EngineDetailsView()
            }
        }
    }
}

// MARK: - Engine Health Card
struct EngineHealthCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(spacing: 15) {
            HStack {
                Image(systemName: "engine.combustion.fill")
                    .font(.title)
                    .foregroundColor(vehicleData.engineData?.statusColor ?? .gray)
                
                VStack(alignment: .leading) {
                    Text("Engine Health")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text(vehicleData.engineData?.condition ?? "Unknown")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(vehicleData.engineData?.statusColor ?? .gray)
                }
                
                Spacer()
                
                VStack {
                    Text("\(Int(vehicleData.engineData?.health ?? 0))%")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(vehicleData.engineData?.statusColor ?? .gray)
                    
                    Text("Health Score")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // Health Progress Bar
            ProgressView(value: vehicleData.engineData?.health ?? 0, total: 100)
                .tint(vehicleData.engineData?.statusColor ?? .gray)
                .scaleEffect(x: 1, y: 2, anchor: .center)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 5)
    }
}

// MARK: - Engine Metrics Grid
struct EngineMetricsGrid: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 15) {
            EngineMetricCard(
                title: "RPM",
                value: String(format: "%.0f", vehicleData.engineData?.rpm ?? 0),
                unit: "RPM",
                icon: "speedometer",
                color: rpmColor(vehicleData.engineData?.rpm ?? 0)
            )
            
            EngineMetricCard(
                title: "Oil Pressure",
                value: String(format: "%.1f", vehicleData.engineData?.oilPressure ?? 0),
                unit: "PSI",
                icon: "drop.fill",
                color: pressureColor(vehicleData.engineData?.oilPressure ?? 0)
            )
            
            EngineMetricCard(
                title: "Oil Temperature",
                value: String(format: "%.1f", vehicleData.engineData?.oilTemperature ?? 0),
                unit: "°C",
                icon: "thermometer",
                color: temperatureColor(vehicleData.engineData?.oilTemperature ?? 0)
            )
            
            EngineMetricCard(
                title: "Coolant Temp",
                value: String(format: "%.1f", vehicleData.engineData?.coolantTemperature ?? 0),
                unit: "°C",
                icon: "snowflake",
                color: coolantTempColor(vehicleData.engineData?.coolantTemperature ?? 0)
            )
            
            EngineMetricCard(
                title: "Fuel Pressure",
                value: String(format: "%.1f", vehicleData.engineData?.fuelPressure ?? 0),
                unit: "PSI",
                icon: "fuelpump.fill",
                color: fuelPressureColor(vehicleData.engineData?.fuelPressure ?? 0)
            )
            
            EngineMetricCard(
                title: "Coolant Pressure",
                value: String(format: "%.1f", vehicleData.engineData?.coolantPressure ?? 0),
                unit: "PSI",
                icon: "gauge",
                color: coolantPressureColor(vehicleData.engineData?.coolantPressure ?? 0)
            )
        }
    }
    
    // Color functions for different metrics
    private func rpmColor(_ rpm: Double) -> Color {
        if rpm < 1000 || rpm > 4000 { return .red }
        if rpm > 3000 { return .orange }
        return .green
    }
    
    private func pressureColor(_ pressure: Double) -> Color {
        if pressure < 20 { return .red }
        if pressure < 30 { return .orange }
        return .green
    }
    
    private func temperatureColor(_ temp: Double) -> Color {
        if temp > 100 { return .red }
        if temp > 90 { return .orange }
        return .green
    }
    
    private func coolantTempColor(_ temp: Double) -> Color {
        if temp > 95 { return .red }
        if temp > 85 { return .orange }
        return .green
    }
    
    private func fuelPressureColor(_ pressure: Double) -> Color {
        if pressure < 2.5 { return .red }
        if pressure < 3.0 { return .orange }
        return .green
    }
    
    private func coolantPressureColor(_ pressure: Double) -> Color {
        if pressure < 1.5 { return .red }
        if pressure < 2.0 { return .orange }
        return .green
    }
}

// MARK: - Engine Metric Card
struct EngineMetricCard: View {
    let title: String
    let value: String
    let unit: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundColor(color)
                
                Spacer()
                
                Circle()
                    .fill(color)
                    .frame(width: 8, height: 8)
            }
            
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                HStack(alignment: .bottom, spacing: 4) {
                    Text(value)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(color)
                    
                    Text(unit)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.bottom, 2)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

// MARK: - Engine Performance Card
struct EnginePerformanceCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Performance Analysis")
                .font(.headline)
                .fontWeight(.semibold)
            
            HStack(spacing: 20) {
                PerformanceIndicator(
                    title: "Power Output",
                    value: calculatePowerOutput(),
                    unit: "%",
                    color: .blue
                )
                
                PerformanceIndicator(
                    title: "Efficiency",
                    value: calculateEfficiency(),
                    unit: "%",
                    color: .green
                )
                
                PerformanceIndicator(
                    title: "Load Factor",
                    value: calculateLoadFactor(),
                    unit: "%",
                    color: .orange
                )
            }
            
            // Performance Status
            HStack {
                Image(systemName: getPerformanceIcon())
                    .foregroundColor(getPerformanceColor())
                
                Text(getPerformanceStatus())
                    .font(.subheadline)
                    .foregroundColor(getPerformanceColor())
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 3)
    }
    
    private func calculatePowerOutput() -> Double {
        guard let engine = vehicleData.engineData else { return 0 }
        return min(100, (engine.rpm / 3000.0) * 100)
    }
    
    private func calculateEfficiency() -> Double {
        guard let engine = vehicleData.engineData else { return 0 }
        let tempFactor = max(0, min(1, (100 - abs(engine.oilTemperature - 80)) / 20))
        let pressureFactor = max(0, min(1, engine.oilPressure / 50))
        return (tempFactor + pressureFactor) * 50
    }
    
    private func calculateLoadFactor() -> Double {
        guard let engine = vehicleData.engineData else { return 0 }
        return min(100, (engine.rpm / 4000.0) * 100)
    }
    
    private func getPerformanceStatus() -> String {
        let efficiency = calculateEfficiency()
        if efficiency >= 80 { return "Optimal Performance" }
        if efficiency >= 60 { return "Good Performance" }
        if efficiency >= 40 { return "Fair Performance" }
        return "Poor Performance"
    }
    
    private func getPerformanceColor() -> Color {
        let efficiency = calculateEfficiency()
        if efficiency >= 80 { return .green }
        if efficiency >= 60 { return .blue }
        if efficiency >= 40 { return .orange }
        return .red
    }
    
    private func getPerformanceIcon() -> String {
        let efficiency = calculateEfficiency()
        if efficiency >= 80 { return "checkmark.circle.fill" }
        if efficiency >= 60 { return "info.circle.fill" }
        if efficiency >= 40 { return "exclamationmark.triangle.fill" }
        return "xmark.circle.fill"
    }
}

// MARK: - Performance Indicator
struct PerformanceIndicator: View {
    let title: String
    let value: Double
    let unit: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text("\(Int(value))\(unit)")
                .font(.headline)
                .fontWeight(.bold)
                .foregroundColor(color)
            
            ProgressView(value: value, total: 100)
                .tint(color)
                .frame(height: 4)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Engine Alerts Card
struct EngineAlertsCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Engine Alerts")
                .font(.headline)
                .fontWeight(.semibold)
            
            let engineAlerts = vehicleData.maintenanceAlerts.filter { alert in
                alert.component.lowercased().contains("engine") ||
                alert.component.lowercased().contains("oil") ||
                alert.component.lowercased().contains("coolant") ||
                alert.component.lowercased().contains("fuel")
            }
            
            if engineAlerts.isEmpty {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("No engine alerts")
                        .foregroundColor(.secondary)
                }
                .padding()
            } else {
                ForEach(engineAlerts.prefix(3)) { alert in
                    MaintenanceAlertRow(alert: alert)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

// MARK: - Engine Details View
struct EngineDetailsView: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Detailed Engine Information
                    DetailedEngineInfo()
                    
                    // Historical Performance (placeholder)
                    HistoricalPerformanceCard()
                    
                    // Maintenance History (placeholder)
                    MaintenanceHistoryCard()
                }
                .padding()
            }
            .navigationTitle("Engine Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Detailed Engine Info
struct DetailedEngineInfo: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Detailed Engine Information")
                .font(.headline)
                .fontWeight(.semibold)
            
            if let engine = vehicleData.engineData {
                VStack(spacing: 10) {
                    DetailRow(label: "Engine RPM", value: "\(Int(engine.rpm))", unit: "RPM")
                    DetailRow(label: "Oil Pressure", value: String(format: "%.2f", engine.oilPressure), unit: "PSI")
                    DetailRow(label: "Fuel Pressure", value: String(format: "%.2f", engine.fuelPressure), unit: "PSI")
                    DetailRow(label: "Coolant Pressure", value: String(format: "%.2f", engine.coolantPressure), unit: "PSI")
                    DetailRow(label: "Oil Temperature", value: String(format: "%.1f", engine.oilTemperature), unit: "°C")
                    DetailRow(label: "Coolant Temperature", value: String(format: "%.1f", engine.coolantTemperature), unit: "°C")
                    DetailRow(label: "Engine Health", value: String(format: "%.1f", engine.health), unit: "%")
                    DetailRow(label: "Last Updated", value: DateFormatter.timeFormatter.string(from: engine.timestamp), unit: "")
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 3)
    }
}

// MARK: - Detail Row
struct DetailRow: View {
    let label: String
    let value: String
    let unit: String
    
    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            Spacer()
            
            HStack(spacing: 4) {
                Text(value)
                    .font(.subheadline)
                    .fontWeight(.medium)
                if !unit.isEmpty {
                    Text(unit)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Historical Performance Card
struct HistoricalPerformanceCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Performance Trends")
                .font(.headline)
                .fontWeight(.semibold)
            
            Text("24-hour performance summary")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            // Placeholder for charts
            HStack(spacing: 20) {
                VStack {
                    Text("Avg RPM")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("2,450")
                        .font(.headline)
                        .fontWeight(.bold)
                }
                
                VStack {
                    Text("Peak RPM")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("3,200")
                        .font(.headline)
                        .fontWeight(.bold)
                }
                
                VStack {
                    Text("Efficiency")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("87%")
                        .font(.headline)
                        .fontWeight(.bold)
                        .foregroundColor(.green)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

// MARK: - Maintenance History Card
struct MaintenanceHistoryCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Recent Maintenance")
                .font(.headline)
                .fontWeight(.semibold)
            
            VStack(spacing: 10) {
                MaintenanceHistoryRow(date: "Nov 5, 2024", item: "Oil Change", status: "Completed")
                MaintenanceHistoryRow(date: "Oct 15, 2024", item: "Air Filter", status: "Completed")
                MaintenanceHistoryRow(date: "Sep 20, 2024", item: "Spark Plugs", status: "Completed")
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 3)
    }
}

// MARK: - Maintenance History Row
struct MaintenanceHistoryRow: View {
    let date: String
    let item: String
    let status: String
    
    var body: some View {
        HStack {
            VStack(alignment: .leading) {
                Text(item)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text(date)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(status)
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color.green.opacity(0.2))
                .foregroundColor(.green)
                .cornerRadius(8)
        }
    }
}

// MARK: - Extensions
extension DateFormatter {
    static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.timeStyle = .medium
        return formatter
    }()
}

// MARK: - Preview
struct EngineStatusView_Previews: PreviewProvider {
    static var previews: some View {
        EngineStatusView()
            .environmentObject(NetworkManager.shared)
            .environmentObject(VehicleDataModel.sampleData())
    }
}
