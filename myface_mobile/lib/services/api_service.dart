import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http/io_client.dart';
import 'package:myface_mobile/config/app_config.dart';
import 'package:myface_mobile/models/user_profile.dart';
import 'package:myface_mobile/models/log_entry.dart';

class ApiService {
  ApiService._privateConstructor();
  static final ApiService instance = ApiService._privateConstructor();

  /// Creates a secure HttpClient based on current environment.
  /// 
  /// In Production, strictly enforces verified Root CAs certificates.
  /// In Development, permits self-signed certificate overrides for local IP networks.
  IOClient _getClient() {
    final HttpClient innerClient = HttpClient();
    final env = AppConfig.instance.environment;

    if (env == AppEnvironment.development) {
      innerClient.badCertificateCallback = (X509Certificate cert, String host, int port) => true;
    } else {
      innerClient.badCertificateCallback = (X509Certificate cert, String host, int port) {
        // Enforce strict TLS certificate pinning / chain checks in production
        return false;
      };
    }
    return IOClient(innerClient);
  }

  /// Evaluates device internet connectivity.
  Future<bool> isOnline() async {
    try {
      final result = await InternetAddress.lookup('google.com')
          .timeout(const Duration(seconds: 4));
      return result.isNotEmpty && result[0].rawAddress.isNotEmpty;
    } catch (_) {
      return false;
    }
  }

  /// Standardized request execution wrapper integrating Offline Detection, Timeouts,
  /// and Exponential Backoff Retries.
  Future<http.Response> _sendRequest(Future<http.Response> Function(IOClient client) requestFn) async {
    // 1. Offline pre-check
    final online = await isOnline();
    if (!online) {
      throw const SocketException("Device is offline. Check connection settings.");
    }

    final client = _getClient();
    int attempt = 0;
    int backoffMs = 500;

    try {
      while (true) {
        try {
          attempt++;
          final response = await requestFn(client).timeout(
            const Duration(seconds: AppConfig.requestTimeoutSeconds),
          );
          return response;
        } catch (e) {
          if (attempt >= AppConfig.maxRetryAttempts) {
            rethrow; // Out of retries, propagate error
          }
          // Sleep with exponential backoff delay before retrying
          await Future.delayed(Duration(milliseconds: backoffMs));
          backoffMs *= 2;
        }
      }
    } finally {
      client.close();
    }
  }

  /// Pings backend healthcheck endpoint.
  Future<bool> checkHealth() async {
    try {
      final res = await _sendRequest((client) => client.get(
        Uri.parse('${AppConfig.instance.baseUrl}/health'),
      ));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        return data['status'] == 'healthy';
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  /// Fetches enrolled user profile summaries list.
  Future<List<UserProfile>> fetchUsers() async {
    final res = await _sendRequest((client) => client.get(
      Uri.parse('${AppConfig.instance.baseUrl}/users'),
    ));

    if (res.statusCode == 200) {
      final Map<String, dynamic> data = jsonDecode(res.body);
      final List<dynamic> list = data['users'] ?? [];
      return list.map((json) => UserProfile.fromJson(json)).toList();
    } else {
      throw HttpException('Failed to load user list. Status: ${res.statusCode}');
    }
  }

  /// Fetches system authentication attempt audit logs.
  Future<List<LogEntry>> fetchLogs() async {
    final res = await _sendRequest((client) => client.get(
      Uri.parse('${AppConfig.instance.baseUrl}/logs'),
    ));

    if (res.statusCode == 200) {
      final Map<String, dynamic> data = jsonDecode(res.body);
      final List<dynamic> list = data['logs'] ?? [];
      return list.map((json) => LogEntry.fromJson(json)).toList();
    } else {
      throw HttpException('Failed to load logs. Status: ${res.statusCode}');
    }
  }

  /// Fetches system diagnostics performance metrics (EER/AUC).
  Future<Map<String, dynamic>> fetchMetrics() async {
    final res = await _sendRequest((client) => client.get(
      Uri.parse('${AppConfig.instance.baseUrl}/metrics'),
    ));

    if (res.statusCode == 200) {
      return jsonDecode(res.body) as Map<String, dynamic>;
    } else {
      throw HttpException('Failed to load metrics. Status: ${res.statusCode}');
    }
  }

  /// Enrolls a new user with multiple face snapshot images.
  Future<Map<String, dynamic>> enrollUser(String username, List<File> imageFiles) async {
    final online = await isOnline();
    if (!online) {
      throw const SocketException("Device is offline. Check connection settings.");
    }

    final client = _getClient();
    final uri = Uri.parse('${AppConfig.instance.baseUrl}/enroll');
    final request = http.MultipartRequest('POST', uri);
    request.fields['username'] = username;

    for (var file in imageFiles) {
      request.files.add(await http.MultipartFile.fromPath('files', file.path));
    }

    try {
      final streamResponse = await client.send(request).timeout(
        const Duration(seconds: AppConfig.requestTimeoutSeconds * 2), // Extra time for upload + training
      );
      final res = await http.Response.fromStream(streamResponse);
      
      if (res.statusCode == 201) {
        return jsonDecode(res.body) as Map<String, dynamic>;
      } else {
        final errorMsg = jsonDecode(res.body)['detail'] ?? 'Enrollment failed.';
        throw HttpException(errorMsg.toString());
      }
    } finally {
      client.close();
    }
  }

  /// Authenticates user credentials using a captured snapshot.
  Future<Map<String, dynamic>> authenticateUser(String userId, File imageFile) async {
    final online = await isOnline();
    if (!online) {
      throw const SocketException("Device is offline. Check connection settings.");
    }

    final client = _getClient();
    final uri = Uri.parse('${AppConfig.instance.baseUrl}/authenticate');
    final request = http.MultipartRequest('POST', uri);
    request.fields['user_id'] = userId;
    request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

    try {
      final streamResponse = await client.send(request).timeout(
        const Duration(seconds: AppConfig.requestTimeoutSeconds),
      );
      final res = await http.Response.fromStream(streamResponse);
      
      if (res.statusCode == 200) {
        return jsonDecode(res.body) as Map<String, dynamic>;
      } else {
        final errorMsg = jsonDecode(res.body)['detail'] ?? 'Verification failed.';
        throw HttpException(errorMsg.toString());
      }
    } finally {
      client.close();
    }
  }
}
