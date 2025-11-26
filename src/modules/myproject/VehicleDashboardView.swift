import SwiftUI

struct VehicleDashboardView: View {
    @StateObject private var dataService = VehicleDataService.shared

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Connection Status
                    connectionStatusView

                    if let data = dataService.vehicleData {
                        // Overall Health Card
                        overallHealthCard(data: data)

                        // Engine Status Card
                        engineStatusCard(data: data)

                        // Battery Status Card
                        batteryStatusCard(data: data)

                        // Parts Health
                        partsHealthCard(data: data)

                        // Recommendations
                        recommendationsCard(data: data)

                    } else {
                        waitingForDataView
                    }
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("🚗 Vehicle Monitor")
            .onAppear {
                dataService.startPolling()
            }
            .onDisappear {
                dataService.stopPolling()
            }
            .refreshable {
                // Pull to refresh (short pause to show refresh UI)
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
    }

    // MARK: - Connection Status
    private var connectionStatusView: some View {
        HStack {
            Circle()
                .fill(dataService.isConnected ? Color.green : Color.red)
                .frame(width: 12, height: 12)

            Text(dataService.isConnected ? "Connected" : "Disconnected")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()

            if let lastUpdate = dataService.lastUpdateTime {
                Text("Updated: \(lastUpdate, formatter: timeFormatter)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.horizontal)
    }

    // MARK: - Overall Health Card
    private func overallHealthCard(data: VehicleAPIResponse) -> some View {
        VStack(spacing: 12) {
            Text("Overall Vehicle Health")
                .font(.headline)

            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.3), lineWidth: 15)
                    .frame(width: 120, height: 120)

                Circle()
                    .trim(from: 0, to: CGFloat(max(0.0, min(1.0, data.overall.vehicleHealth / 100.0))))
                    .stroke(healthColor(data.overall.vehicleHealth), lineWidth: 15)
                    .frame(width: 120, height: 120)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut(duration: 0.5), value: data.overall.vehicleHealth)

                VStack {
                    Text("\(Int(data.overall.vehicleHealth))%")
                        .font(.title)
                        .fontWeight(.bold)
                    Text(data.overall.status)
                        .font(.caption)
                        .foregroundColor(statusColor(data.overall.status))
                }
            }

            HStack(spacing: 30) {
                VStack {
                    Text("🚗 Engine")
                        .font(.caption)
                    Text("\(Int(data.overall.engineHealth))%")
                        .fontWeight(.semibold)
                        .foregroundColor(healthColor(data.overall.engineHealth))
                }

                VStack {
                    Text("🔋 Battery")
                        .font(.caption)
                    Text("\(Int(data.overall.batteryHealth))%")
                        .fontWeight(.semibold)
                        .foregroundColor(healthColor(data.overall.batteryHealth))
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    // MARK: - Engine Status Card
    private func engineStatusCard(data: VehicleAPIResponse) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("🚗 Engine Status")
                    .font(.headline)
                Spacer()
                Text(data.engine.condition.uppercased())
                    .font(.caption)
                    .fontWeight(.bold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(conditionColor(data.engine.condition))
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }

            // Health Score Bar
            VStack(alignment: .leading, spacing: 4) {
                Text("Health Score: \(Int(data.engine.healthScore))%")
                    .font(.caption)
                    .foregroundColor(.secondary)

                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Rectangle()
                            .fill(Color.gray.opacity(0.3))
                            .frame(height: 8)
                            .cornerRadius(4)

                        Rectangle()
                            .fill(healthColor(data.engine.healthScore))
                            .frame(width: geometry.size.width * CGFloat(max(0.0, min(1.0, data.engine.healthScore / 100.0))), height: 8)
                            .cornerRadius(4)
                            .animation(.easeInOut(duration: 0.5), value: data.engine.healthScore)
                    }
                }
                .frame(height: 8)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                sensorValueView(title: "RPM", value: "\(Int(data.engine.sensorData.rpm))", icon: "gauge")
                sensorValueView(title: "Oil Pressure", value: String(format: "%.1f psi", data.engine.sensorData.lubOilPressure), icon: "drop.fill")
                sensorValueView(title: "Fuel Pressure", value: String(format: "%.1f psi", data.engine.sensorData.fuelPressure), icon: "fuelpump.fill")
                sensorValueView(title: "Coolant Pressure", value: String(format: "%.1f psi", data.engine.sensorData.coolantPressure), icon: "wind")
                sensorValueView(title: "Oil Temp", value: String(format: "%.1f°C", data.engine.sensorData.lubOilTemp), icon: "thermometer")
                sensorValueView(title: "Coolant Temp", value: String(format: "%.1f°C", data.engine.sensorData.coolantTemp), icon: "thermometer.sun")
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    // MARK: - Battery Status Card
    private func batteryStatusCard(data: VehicleAPIResponse) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("🔋 Battery Status")
                    .font(.headline)
                Spacer()
                Text(data.battery.priorityLevel)
                    .font(.caption)
                    .fontWeight(.bold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(priorityColor(data.battery.priorityLevel))
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }

            // Safety Warning
            if !data.battery.safetyWarning.isEmpty {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.red)
                    Text(data.battery.safetyWarning)
                        .font(.caption)
                        .foregroundColor(.red)
                }
                .padding(8)
                .background(Color.red.opacity(0.1))
                .cornerRadius(8)
            }

            // Health Score Bar
            VStack(alignment: .leading, spacing: 4) {
                Text("Health Score: \(Int(data.battery.healthScore))%")
                    .font(.caption)
                    .foregroundColor(.secondary)

                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Rectangle()
                            .fill(Color.gray.opacity(0.3))
                            .frame(height: 8)
                            .cornerRadius(4)

                        Rectangle()
                            .fill(healthColor(data.battery.healthScore))
                            .frame(width: geometry.size.width * CGFloat(max(0.0, min(1.0, data.battery.healthScore / 100.0))), height: 8)
                            .cornerRadius(4)
                            .animation(.easeInOut(duration: 0.5), value: data.battery.healthScore)
                    }
                }
                .frame(height: 8)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                sensorValueView(title: "Voltage", value: String(format: "%.2fV", data.battery.sensorData.voltage), icon: "bolt.fill")
                sensorValueView(title: "Current", value: String(format: "%.2fA", data.battery.sensorData.current), icon: "arrow.right")
                sensorValueView(title: "Temperature", value: String(format: "%.1f°C", data.battery.sensorData.temperature), icon: "thermometer")
                sensorValueView(title: "Capacity", value: String(format: "%.1f%%", data.battery.sensorData.capacity), icon: "battery.75")
                sensorValueView(title: "Resistance", value: String(format: "%.3fΩ", data.battery.sensorData.internalResistance), icon: "waveform.path")
                sensorValueView(title: "Cycles", value: "\(Int(data.battery.sensorData.chargeCycles))", icon: "arrow.triangle.2.circlepath")
            }

            // Remaining Life
            HStack {
                Image(systemName: "clock")
                    .foregroundColor(.secondary)
                Text("Remaining Life: \(data.battery.remainingDays) days (~\(String(format: "%.1f", data.battery.remainingMonths)) months)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Data Source
            HStack {
                Image(systemName: data.battery.isRealData ? "antenna.radiowaves.left.and.right" : "arrow.triangle.2.circlepath")
                    .foregroundColor(.secondary)
                Text("Data Source: \(data.battery.isRealData ? "Real Sensors" : "Simulated")")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    // MARK: - Parts Health Card
    private func partsHealthCard(data: VehicleAPIResponse) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("🔧 Engine Parts Health")
                .font(.headline)

            ForEach(Array(data.engine.partsHealth.keys.sorted()), id: \.self) { part in
                if let health = data.engine.partsHealth[part] {
                    HStack {
                        Image(systemName: partIcon(for: part))
                            .foregroundColor(healthColor(health))
                            .frame(width: 20)

                        Text(formatPartName(part))
                            .font(.caption)
                            .frame(width: 100, alignment: .leading)

                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                Rectangle()
                                    .fill(Color.gray.opacity(0.3))
                                    .frame(height: 8)
                                    .cornerRadius(4)

                                Rectangle()
                                    .fill(healthColor(health))
                                    .frame(width: geometry.size.width * CGFloat(max(0.0, min(1.0, health / 100.0))), height: 8)
                                    .cornerRadius(4)
                                    .animation(.easeInOut(duration: 0.5), value: health)
                            }
                        }
                        .frame(height: 8)

                        Text("\(Int(health))%")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(healthColor(health))
                            .frame(width: 45, alignment: .trailing)

                        Image(systemName: healthIcon(health))
                            .foregroundColor(healthColor(health))
                            .frame(width: 20)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    // MARK: - Recommendations Card
    private func recommendationsCard(data: VehicleAPIResponse) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("📋 Maintenance Recommendations")
                .font(.headline)

            if data.recommendations.engine.isEmpty {
                HStack {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("No maintenance required at this time")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } else {
                ForEach(data.recommendations.engine.prefix(5)) { rec in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Circle()
                                .fill(priorityColor(rec.priorityLevel))
                                .frame(width: 10, height: 10)

                            Text(rec.part)
                                .font(.subheadline)
                                .fontWeight(.semibold)

                            Spacer()

                            Text(rec.currentHealth)
                                .font(.caption)
                                .foregroundColor(healthColor(rec.healthScore))
                        }

                        HStack {
                            Image(systemName: "road.lanes")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text("Replace in: \(rec.remainingKmFormatted)")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Spacer()

                            Image(systemName: "calendar")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text(rec.recommendedReplacementDate)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Text(rec.priority)
                            .font(.caption2)
                            .foregroundColor(priorityColor(rec.priorityLevel))
                    }
                    .padding(10)
                    .background(priorityColor(rec.priorityLevel).opacity(0.1))
                    .cornerRadius(8)
                }
            }

            // Battery Recommendations
            if let batteryRec = data.recommendations.battery.first {
                Divider()

                Text("🔋 Battery Recommendation")
                    .font(.subheadline)
                    .fontWeight(.semibold)

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Circle()
                            .fill(priorityColor(batteryRec.priorityLevel))
                            .frame(width: 10, height: 10)

                        Text(batteryRec.component)
                            .font(.subheadline)
                            .fontWeight(.semibold)

                        Spacer()

                        Text(batteryRec.currentHealth)
                            .font(.caption)
                            .foregroundColor(healthColor(batteryRec.healthScore))
                    }

                    Text(batteryRec.priority)
                        .font(.caption)
                        .foregroundColor(priorityColor(batteryRec.priorityLevel))

                    HStack {
                        Image(systemName: "calendar")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("Replace by: \(batteryRec.recommendedReplacementDate)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding(10)
                .background(priorityColor(batteryRec.priorityLevel).opacity(0.1))
                .cornerRadius(8)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    // MARK: - Waiting View
    private var waitingForDataView: some View {
        VStack(spacing: 20) {
            Image(systemName: "car.fill")
                .font(.system(size: 60))
                .foregroundColor(.gray)

            ProgressView()
                .scaleEffect(1.5)

            Text("Waiting for vehicle data...")
                .font(.headline)
                .foregroundColor(.secondary)

            Text("Make sure the backend server is running")
                .font(.caption)
                .foregroundColor(.secondary)

            if let error = dataService.connectionError {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text(error)
                        .font(.caption)
                        .foregroundColor(.red)
                }
                .padding()
                .background(Color.red.opacity(0.1))
                .cornerRadius(8)
            }

            Button(action: {
                dataService.stopPolling()
                dataService.startPolling()
            }) {
                Label("Retry Connection", systemImage: "arrow.clockwise")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
        }
        .padding(50)
    }

    // MARK: - Helper Views
    private func sensorValueView(title: String, value: String, icon: String) -> some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 20)

            VStack(alignment: .leading) {
                Text(title)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text(value)
                    .font(.caption)
                    .fontWeight(.semibold)
            }

            Spacer()
        }
        .padding(8)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(8)
    }

    // MARK: - Helper Functions
    private func healthColor(_ health: Double) -> Color {
        if health > 70 { return .green }
        if health > 40 { return .orange }
        return .red
    }

    private func conditionColor(_ condition: String) -> Color {
        switch condition.lowercased() {
        case "good": return .green
        case "moderate": return .orange
        case "bad": return .red
        default: return .gray
        }
    }

    private func statusColor(_ status: String) -> Color {
        switch status.uppercased() {
        case "NORMAL": return .green
        case "WARNING": return .orange
        case "CRITICAL": return .red
        default: return .gray
        }
    }

    private func priorityColor(_ priority: String) -> Color {
        switch priority.uppercased() {
        case "LOW": return .green
        case "MEDIUM": return .orange
        case "HIGH": return .orange
        case "CRITICAL": return .red
        default: return .gray
        }
    }

    private func healthIcon(_ health: Double) -> String {
        if health > 70 { return "checkmark.circle.fill" }
        if health > 40 { return "exclamationmark.triangle.fill" }
        return "xmark.circle.fill"
    }

    private func partIcon(for part: String) -> String {
        switch part.lowercased() {
        case "oil_pump": return "drop.fill"
        case "oil_filter": return "line.3.horizontal.decrease.circle"
        case "coolant_pump": return "wind"
        case "thermostat": return "thermometer"
        case "fuel_pump": return "fuelpump.fill"
        case "fuel_filter": return "line.3.horizontal.decrease"
        default: return "gear"
        }
    }

    private func formatPartName(_ name: String) -> String {
        name.replacingOccurrences(of: "_", with: " ").capitalized
    }

    private var timeFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.timeStyle = .medium
        return formatter
    }
}

// MARK: - Preview
#Preview {
    VehicleDashboardView()
}
