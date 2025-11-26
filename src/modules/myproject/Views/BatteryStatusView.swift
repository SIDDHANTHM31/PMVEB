import SwiftUI

struct BatteryStatusView: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    @EnvironmentObject var networkManager: NetworkManager
    @State private var showingDetails = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Battery Health Overview
                    BatteryHealthCard()
                    
                    // Battery Metrics
                    BatteryMetricsGrid()
                    
                    // Battery Performance Analysis
                    BatteryPerformanceCard()
                    
                    // Battery Alerts
                    BatteryAlertsCard()
                }
                .padding()
            }
            .navigationTitle("Battery Status")
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
                BatteryDetailsView()
            }
        }
    }
}

// MARK: - Battery Health Card
struct BatteryHealthCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(spacing: 15) {
            HStack {
                Image(systemName: "battery.100.bolt")
                    .font(.title)
                    .foregroundColor(vehicleData.batteryData?.statusColor ?? .gray)
                
                VStack(alignment: .leading) {
                    Text("Battery Health")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text(vehicleData.batteryData?.condition ?? "Unknown")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(vehicleData.batteryData?.statusColor ?? .gray)
                }
                
                Spacer()
                
                VStack {
                    Text("\(Int(vehicleData.batteryData?.health ?? 0))%")
                        .font(.title)
                        .fontWeight(.bold)
                        .foregroundColor(vehicleData.batteryData?.statusColor ?? .gray)
                    
                    Text("Health Score")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            // Health Progress Bar
            ProgressView(value: vehicleData.batteryData?.health ?? 0, total: 100)
                .tint(vehicleData.batteryData?.statusColor ?? .gray)
                .scaleEffect(x: 1, y: 2, anchor: .center)
            
            // Voltage Status Indicator
            HStack {
                Image(systemName: getBatteryIcon())
                    .foregroundColor(getVoltageColor())
                
                Text("\(String(format: "%.2f", vehicleData.batteryData?.voltage ?? 0))V")
                    .font(.headline)
                    .fontWeight(.semibold)
                
                Spacer()
                
                Text(vehicleData.batteryData?.voltageStatus ?? "Unknown")
                    .font(.subheadline)
                    .foregroundColor(getVoltageColor())
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 5)
    }
    
    private func getBatteryIcon() -> String {
        guard let voltage = vehicleData.batteryData?.voltage else { return "battery.0" }
        
        if voltage >= 12.6 { return "battery.100" }
        else if voltage >= 12.4 { return "battery.75" }
        else if voltage >= 12.0 { return "battery.50" }
        else if voltage >= 11.5 { return "battery.25" }
        else { return "battery.0" }
    }
    
    private func getVoltageColor() -> Color {
        guard let voltage = vehicleData.batteryData?.voltage else { return .gray }
        
        if voltage >= 12.4 { return .green }
        else if voltage >= 12.0 { return .orange }
        else { return .red }
    }
}

// MARK: - Battery Metrics Grid
struct BatteryMetricsGrid: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 15) {
            BatteryMetricCard(
                title: "Voltage",
                value: String(format: "%.2f", vehicleData.batteryData?.voltage ?? 0),
                unit: "V",
                icon: "bolt.fill",
                color: voltageColor(vehicleData.batteryData?.voltage ?? 0)
            )
            
            BatteryMetricCard(
                title: "Current",
                value: String(format: "%.1f", vehicleData.batteryData?.current ?? 0),
                unit: "A",
                icon: "arrow.right.circle.fill",
                color: currentColor(vehicleData.batteryData?.current ?? 0)
            )
            
            BatteryMetricCard(
                title: "Temperature",
                value: String(format: "%.1f", vehicleData.batteryData?.temperature ?? 0),
                unit: "°C",
                icon: "thermometer",
                color: temperatureColor(vehicleData.batteryData?.temperature ?? 0)
            )
            
            BatteryMetricCard(
                title: "Capacity",
                value: String(format: "%.1f", vehicleData.batteryData?.capacity ?? 0),
                unit: "%",
                icon: "battery.75",
                color: capacityColor(vehicleData.batteryData?.capacity ?? 0)
            )
            
            BatteryMetricCard(
                title: "Resistance",
                value: String(format: "%.3f", vehicleData.batteryData?.internalResistance ?? 0),
                unit: "Ω",
                icon: "resistance",
                color: resistanceColor(vehicleData.batteryData?.internalResistance ?? 0)
            )
            
            BatteryMetricCard(
                title: "Charge Cycles",
                value: "\(vehicleData.batteryData?.chargeCycles ?? 0)",
                unit: "cycles",
                icon: "arrow.triangle.2.circlepath",
                color: cyclesColor(vehicleData.batteryData?.chargeCycles ?? 0)
            )
        }
    }
    
    // Color functions for different metrics
    private func voltageColor(_ voltage: Double) -> Color {
        if voltage >= 12.4 { return .green }
        if voltage >= 12.0 { return .orange }
        return .red
    }
    
    private func currentColor(_ current: Double) -> Color {
        let absCurrent = abs(current)
        if absCurrent <= 5 { return .green }
        if absCurrent <= 10 { return .orange }
        return .red
    }
    
    private func temperatureColor(_ temp: Double) -> Color {
        if temp >= 0 && temp <= 35 { return .green }
        if temp <= 45 { return .orange }
        return .red
    }
    
    private func capacityColor(_ capacity: Double) -> Color {
        if capacity >= 80 { return .green }
        if capacity >= 60 { return .orange }
        return .red
    }
    
    private func resistanceColor(_ resistance: Double) -> Color {
        if resistance <= 0.1 { return .green }
        if resistance <= 0.2 { return .orange }
        return .red
    }
    
    private func cyclesColor(_ cycles: Int) -> Color {
        if cycles < 500 { return .green }
        if cycles < 1000 { return .orange }
        return .red
    }
}

// MARK: - Battery Metric Card
struct BatteryMetricCard: View {
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

// MARK: - Battery Performance Card
struct BatteryPerformanceCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Battery Performance")
                .font(.headline)
                .fontWeight(.semibold)
            
            HStack(spacing: 20) {
                BatteryPerformanceIndicator(
                    title: "Efficiency",
                    value: calculateEfficiency(),
                    unit: "%",
                    color: .blue
                )
                
                BatteryPerformanceIndicator(
                    title: "Life Remaining",
                    value: calculateLifeRemaining(),
                    unit: "%",
                    color: .green
                )
                
                BatteryPerformanceIndicator(
                    title: "Charge Rate",
                    value: calculateChargeRate(),
                    unit: "%",
                    color: .orange
                )
            }
            
            // Performance Status
            HStack {
                Image(systemName: getBatteryPerformanceIcon())
                    .foregroundColor(getBatteryPerformanceColor())
                
                Text(getBatteryPerformanceStatus())
                    .font(.subheadline)
                    .foregroundColor(getBatteryPerformanceColor())
            }
            
            // Estimated remaining life
            HStack {
                Text("Estimated Life Remaining:")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text(getEstimatedLifeText())
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(getBatteryPerformanceColor())
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(15)
        .shadow(radius: 3)
    }
    
    private func calculateEfficiency() -> Double {
        guard let battery = vehicleData.batteryData else { return 0 }
        let voltageEfficiency = min(100, (battery.voltage / 12.6) * 100)
        let capacityEfficiency = battery.capacity
        return (voltageEfficiency + capacityEfficiency) / 2
    }
    
    private func calculateLifeRemaining() -> Double {
        guard let battery = vehicleData.batteryData else { return 0 }
        let cycleLife = max(0, 100 - (Double(battery.chargeCycles) / 10))
        let healthLife = battery.health
        return min(cycleLife, healthLife)
    }
    
    private func calculateChargeRate() -> Double {
        guard let battery = vehicleData.batteryData else { return 0 }
        return max(0, min(100, abs(battery.current) * 20))
    }
    
    private func getBatteryPerformanceStatus() -> String {
        let efficiency = calculateEfficiency()
        if efficiency >= 90 { return "Excellent Performance" }
        if efficiency >= 70 { return "Good Performance" }
        if efficiency >= 50 { return "Fair Performance" }
        return "Poor Performance"
    }
    
    private func getBatteryPerformanceColor() -> Color {
        let efficiency = calculateEfficiency()
        if efficiency >= 90 { return .green }
        if efficiency >= 70 { return .blue }
        if efficiency >= 50 { return .orange }
        return .red
    }
    
    private func getBatteryPerformanceIcon() -> String {
        let efficiency = calculateEfficiency()
        if efficiency >= 90 { return "checkmark.circle.fill" }
        if efficiency >= 70 { return "info.circle.fill" }
        if efficiency >= 50 { return "exclamationmark.triangle.fill" }
        return "xmark.circle.fill"
    }
    
    private func getEstimatedLifeText() -> String {
        guard let battery = vehicleData.batteryData else { return "Unknown" }
        let cycles = battery.chargeCycles
        let health = battery.health
        
        if health >= 80 && cycles < 300 {
            return "2-3 years"
        } else if health >= 60 && cycles < 600 {
            return "1-2 years"
        } else if health >= 40 {
            return "6-12 months"
        } else {
            return "Replace soon"
        }
    }
}

// MARK: - Battery Performance Indicator
struct BatteryPerformanceIndicator: View {
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

// MARK: - Battery Alerts Card
struct BatteryAlertsCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Battery Alerts")
                .font(.headline)
                .fontWeight(.semibold)
            
            let batteryAlerts = vehicleData.maintenanceAlerts.filter { alert in
                alert.component.lowercased().contains("battery")
            }
            
            if batteryAlerts.isEmpty {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("No battery alerts")
                        .foregroundColor(.secondary)
                }
                .padding()
            } else {
                ForEach(batteryAlerts.prefix(3)) { alert in
                    MaintenanceAlertRow(alert: alert)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
}

// MARK: - Battery Details View
struct BatteryDetailsView: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Detailed Battery Information
                    DetailedBatteryInfo()
                    
                    // Battery Chemistry Information
                    BatteryChemistryCard()
                    
                    // Maintenance Tips
                    BatteryMaintenanceTipsCard()
                }
                .padding()
            }
            .navigationTitle("Battery Details")
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

// MARK: - Detailed Battery Info
struct DetailedBatteryInfo: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Detailed Battery Information")
                .font(.headline)
                .fontWeight(.semibold)
            
            if let battery = vehicleData.batteryData {
                VStack(spacing: 10) {
                    DetailRow(label: "Voltage", value: String(format: "%.3f", battery.voltage), unit: "V")
                    DetailRow(label: "Current", value: String(format: "%.2f", battery.current), unit: "A")
                    DetailRow(label: "Temperature", value: String(format: "%.1f", battery.temperature), unit: "°C")
                    DetailRow(label: "Capacity", value: String(format: "%.1f", battery.capacity), unit: "%")
                    DetailRow(label: "Internal Resistance", value: String(format: "%.4f", battery.internalResistance), unit: "Ω")
                    DetailRow(label: "Charge Cycles", value: "\(battery.chargeCycles)", unit: "cycles")
                    DetailRow(label: "Battery Health", value: String(format: "%.1f", battery.health), unit: "%")
                    DetailRow(label: "Power Output", value: String(format: "%.1f", battery.voltage * abs(battery.current)), unit: "W")
                    DetailRow(label: "Last Updated", value: DateFormatter.timeFormatter.string(from: battery.timestamp), unit: "")
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 3)
    }
}

// MARK: - Battery Chemistry Card
struct BatteryChemistryCard: View {
    @EnvironmentObject var vehicleData: VehicleDataModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Battery Information")
                .font(.headline)
                .fontWeight(.semibold)
            
            let batteryType = getBatteryType()
            
            VStack(spacing: 10) {
                DetailRow(label: "Battery Type", value: batteryType, unit: "")
                DetailRow(label: "Nominal Voltage", value: getNominalVoltage(), unit: "V")
                DetailRow(label: "Chemistry", value: getChemistry(), unit: "")
                DetailRow(label: "Expected Life", value: getExpectedLife(), unit: "")
                DetailRow(label: "Optimal Temp Range", value: "15-25", unit: "°C")
                DetailRow(label: "Max Charge Cycles", value: getMaxCycles(), unit: "cycles")
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
    
    private func getBatteryType() -> String {
        guard let voltage = vehicleData.batteryData?.voltage else { return "Unknown" }
        return voltage > 6 ? "12V Lead-Acid" : "Li-ion Cell"
    }
    
    private func getNominalVoltage() -> String {
        guard let voltage = vehicleData.batteryData?.voltage else { return "Unknown" }
        return voltage > 6 ? "12.6" : "3.7"
    }
    
    private func getChemistry() -> String {
        guard let voltage = vehicleData.batteryData?.voltage else { return "Unknown" }
        return voltage > 6 ? "Lead-Acid" : "Lithium-ion"
    }
    
    private func getExpectedLife() -> String {
        guard let voltage = vehicleData.batteryData?.voltage else { return "Unknown" }
        return voltage > 6 ? "3-5 years" : "8-10 years"
    }
    
    private func getMaxCycles() -> String {
        guard let voltage = vehicleData.batteryData?.voltage else { return "Unknown" }
        return voltage > 6 ? "300-500" : "2000-5000"
    }
}

// MARK: - Battery Maintenance Tips
struct BatteryMaintenanceTipsCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Maintenance Tips")
                .font(.headline)
                .fontWeight(.semibold)
            
            VStack(alignment: .leading, spacing: 10) {
                MaintenanceTip(
                    icon: "thermometer",
                    title: "Temperature Management",
                    description: "Keep battery temperature between 15-35°C for optimal performance"
                )
                
                MaintenanceTip(
                    icon: "bolt.circle",
                    title: "Charging Habits",
                    description: "Avoid deep discharge cycles and extreme fast charging"
                )
                
                MaintenanceTip(
                    icon: "calendar",
                    title: "Regular Inspection",
                    description: "Check terminals and connections monthly for corrosion"
                )
                
                MaintenanceTip(
                    icon: "gauge",
                    title: "Voltage Monitoring",
                    description: "Monitor voltage regularly; replace if consistently below 12.0V"
                )
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 3)
    }
}

// MARK: - Maintenance Tip
struct MaintenanceTip: View {
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(.blue)
                .frame(width: 24)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

// MARK: - Preview
struct BatteryStatusView_Previews: PreviewProvider {
    static var previews: some View {
        BatteryStatusView()
            .environmentObject(NetworkManager.shared)
            .environmentObject(VehicleDataModel.sampleData())
    }
}
