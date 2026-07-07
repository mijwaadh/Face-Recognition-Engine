class LogEntry {
  final String username;
  final String userId;
  final String timestamp;
  final double livenessScore;
  final double similarityScore;
  final bool authenticated;
  final String status;

  LogEntry({
    required this.username,
    required this.userId,
    required this.timestamp,
    required this.livenessScore,
    required this.similarityScore,
    required this.authenticated,
    required this.status,
  });

  factory LogEntry.fromJson(Map<String, dynamic> json) {
    return LogEntry(
      username: json['username'] as String? ?? '',
      userId: json['user_id'] as String? ?? '',
      timestamp: json['timestamp'] as String? ?? '',
      livenessScore: (json['liveness_score'] as num? ?? 0.0).toDouble(),
      similarityScore: (json['similarity_score'] as num? ?? 0.0).toDouble(),
      authenticated: json['authenticated'] as bool? ?? false,
      status: json['status'] as String? ?? '',
    );
  }
}
