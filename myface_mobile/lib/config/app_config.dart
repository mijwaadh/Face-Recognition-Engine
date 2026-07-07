import 'package:shared_preferences/shared_preferences.dart';

enum AppEnvironment {
  development,
  production,
}

class AppConfig {
  static const String keyBaseUrl = 'base_url';
  static const String keyEnvironment = 'app_environment';
  static const String keyMatchThreshold = 'match_threshold';
  static const String keyLivenessThreshold = 'liveness_threshold';

  // Default production cloud API URL (Removes localhost defaults)
  static const String defaultProdUrl = 'https://biometric-auth-backend.onrender.com';
  static const String defaultDevUrl = 'https://biometric-auth-backend.onrender.com';
  
  static const int requestTimeoutSeconds = 30;
  static const int maxRetryAttempts = 3;

  late SharedPreferences _prefs;

  AppConfig._privateConstructor();
  static final AppConfig instance = AppConfig._privateConstructor();

  /// Initializes SharedPreferences instance.
  Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
  }

  /// Gets current active app environment.
  AppEnvironment get environment {
    final envStr = _prefs.getString(keyEnvironment);
    if (envStr == 'production') {
      return AppEnvironment.production;
    }
    return AppEnvironment.development;
  }

  /// Sets current active app environment.
  Future<bool> setEnvironment(AppEnvironment env) async {
    return await _prefs.setString(keyEnvironment, env.name);
  }

  /// Gets backend API host URL based on environment and custom overrides.
  String get baseUrl {
    final storedUrl = _prefs.getString(keyBaseUrl);
    if (storedUrl != null && 
        storedUrl.isNotEmpty && 
        storedUrl != 'http://10.0.2.2:8000' && 
        storedUrl != 'https://api.myfaceauth.com') {
      return storedUrl;
    }
    return environment == AppEnvironment.production ? defaultProdUrl : defaultDevUrl;
  }

  /// Saves updated backend API host base URL.
  Future<bool> setBaseUrl(String url) async {
    return await _prefs.setString(keyBaseUrl, url);
  }

  /// Gets verification similarity matcher threshold.
  double get matchThreshold => _prefs.getDouble(keyMatchThreshold) ?? 0.80;

  /// Saves verification similarity matcher threshold.
  Future<bool> setMatchThreshold(double value) async {
    return await _prefs.setDouble(keyMatchThreshold, value);
  }

  /// Gets quality check liveness validation threshold.
  double get livenessThreshold => _prefs.getDouble(keyLivenessThreshold) ?? 0.85;

  /// Saves quality check liveness validation threshold.
  Future<bool> setLivenessThreshold(double value) async {
    return await _prefs.setDouble(keyLivenessThreshold, value);
  }
}
