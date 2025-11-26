import SwiftUI

@main
struct VehicleMonitorApp: App {
    // StateObject used by SwiftUI views
    @StateObject private var vehicleData = VehicleDataModel()
    // NetworkManager singleton
    private let networkManager = NetworkManager.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(vehicleData)
                .environmentObject(networkManager)
                .onAppear {
                    // IMPORTANT: set the shared reference BEFORE starting monitoring
                    VehicleDataModel.shared = vehicleData

                    // Optional: debug print to confirm single instance
                    #if DEBUG
                    print("VehicleMonitorApp: VehicleDataModel.shared set ->", VehicleDataModel.shared === vehicleData ? "same instance" : "different instance")
                    #endif

                    // Start polling only after shared is assigned
                    networkManager.startMonitoring()
                }
        }
    }
}
