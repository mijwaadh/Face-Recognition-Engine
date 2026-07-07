class UserProfile {
  final String userId;
  final String username;
  final String enrolledAt;

  UserProfile({
    required this.userId,
    required this.username,
    required this.enrolledAt,
  });

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      userId: json['user_id'] as String? ?? '',
      username: json['username'] as String? ?? '',
      enrolledAt: json['enrolled_at'] as String? ?? '',
    );
  }
}
