import 'package:flutter/material.dart';
import 'package:myface_mobile/config/app_config.dart';

void main() async {
  // Ensure widget bindings are initialized
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize local configurations
  await AppConfig.instance.init();
  
  runApp(const MyFaceMobileApp());
}

class MyFaceMobileApp extends StatelessWidget {
  const MyFaceMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Biometric Face Auth',
      debugShowCheckedModeBanner: false,
      
      // Theme Configuration utilizing Material 3 design directives
      themeMode: ThemeMode.dark,
      darkTheme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xff38bdf8),
          brightness: Brightness.dark,
          background: const Color(0xff0f111a),
          surface: const Color(0xff161c2d),
          primary: const Color(0xff38bdf8),
          secondary: const Color(0xff10b981),
          error: const Color(0xffef4444),
        ),
        
        // Custom styling for premium Material 3 appearance
        cardTheme: const CardTheme(
          color: Color(0x73161c2d),
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(20)),
            side: BorderSide(color: Color(0x14ffffff), width: 1),
          ),
        ),
        
        inputDecorationTheme: const InputDecorationTheme(
          filled: true,
          fillColor: Color(0x0dffffff),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.all(Radius.circular(12)),
            borderSide: BorderSide(color: Color(0x14ffffff)),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.all(Radius.circular(12)),
            borderSide: BorderSide(color: Color(0xff38bdf8)),
          ),
        ),
        
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xff38bdf8),
            foregroundColor: const Color(0xff0f111a),
            padding: const EdgeInsets.symmetric(vertical: 16),
            shape: const RoundedRectangleBorder(
              borderRadius: BorderRadius.all(Radius.circular(12)),
            ),
            textStyle: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
          ),
        ),
      ),
      
      // Placeholder entry screen, to be updated during screen implementations
      home: const TempSetupSplash(),
    );
  }
}

/// Dynamic baseline loading view during project setups.
class TempSetupSplash extends StatefulWidget {
  const TempSetupSplash({super.key});

  @override
  State<TempSetupSplash> createState() => _TempSetupSplashState();
}

class _TempSetupSplashState extends State<TempSetupSplash> {
  @override
  void initState() {
    super.initState();
    loggerCheck();
  }

  void loggerCheck() {
    debugPrint("App Config successfully loaded: Server Host: ${AppConfig.instance.baseUrl}");
  }

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.face_retouching_natural, size: 80, color: Color(0xff38bdf8)),
            SizedBox(height: 24),
            Text(
              'Biometric Pipeline Configured',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.white),
            ),
            SizedBox(height: 8),
            Text(
              'Stage 1: Project Setup Completed Successfully',
              style: TextStyle(fontSize: 14, color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }
}
