import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';
import 'package:myface_mobile/config/app_config.dart';
import 'package:myface_mobile/models/log_entry.dart';
import 'package:myface_mobile/models/user_profile.dart';
import 'package:myface_mobile/services/api_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
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
      home: const MainDashboard(),
    );
  }
}

class MainDashboard extends StatefulWidget {
  const MainDashboard({super.key});

  @override
  State<MainDashboard> createState() => _MainDashboardState();
}

class _MainDashboardState extends State<MainDashboard> {
  int _currentIndex = 0;

  final List<Widget> _tabs = [
    const VerifyTab(),
    const EnrollTab(),
    const AnalyticsTab(),
    const SettingsTab(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            const Icon(Icons.face_retouching_natural, color: Color(0xff38bdf8), size: 28),
            const SizedBox(width: 10),
            Text(
              'Antigravity Face Auth',
              style: TextStyle(
                fontWeight: FontWeight.w800,
                fontSize: 20,
                foreground: Paint()
                  ..shader = const LinearGradient(
                    colors: [Colors.white, Color(0xff38bdf8)],
                  ).createShader(const Rect.fromLTWH(0.0, 0.0, 200.0, 70.0)),
              ),
            ),
          ],
        ),
        backgroundColor: const Color(0xff0f111a),
        elevation: 0,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(
            color: const Color(0x14ffffff),
            height: 1,
          ),
        ),
      ),
      body: IndexedStack(
        index: _currentIndex,
        children: _tabs,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        type: BottomNavigationBarType.fixed,
        backgroundColor: const Color(0xff0f111a),
        selectedItemColor: const Color(0xff38bdf8),
        unselectedItemColor: Colors.grey,
        selectedFontSize: 12,
        unselectedFontSize: 12,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.shield_outlined),
            activeIcon: Icon(Icons.shield, color: Color(0xff38bdf8)),
            label: 'Verify',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person_add_outlined),
            activeIcon: Icon(Icons.person_add, color: Color(0xff38bdf8)),
            label: 'Enroll',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.bar_chart_outlined),
            activeIcon: Icon(Icons.bar_chart, color: Color(0xff38bdf8)),
            label: 'Analytics',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings_outlined),
            activeIcon: Icon(Icons.settings, color: Color(0xff38bdf8)),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}

// ==================== TABS IMPLEMENTATION ====================

/// Tab 1: Verify Face Authentication
class VerifyTab extends StatefulWidget {
  const VerifyTab({super.key});

  @override
  State<VerifyTab> createState() => _VerifyTabState();
}

class _VerifyTabState extends State<VerifyTab> {
  List<UserProfile> _users = [];
  UserProfile? _selectedUser;
  File? _imageFile;
  final ImagePicker _picker = ImagePicker();
  bool _isLoadingUsers = false;
  bool _isVerifying = false;
  
  // Results
  bool? _authSuccess;
  String _authMessage = "";
  double _similarityScore = 0.0;
  double _livenessScore = 0.0;

  @override
  void initState() {
    super.initState();
    _loadUsers();
  }

  Future<void> _loadUsers() async {
    setState(() {
      _isLoadingUsers = true;
    });
    try {
      final list = await ApiService.instance.fetchUsers();
      setState(() {
        _users = list;
        if (_users.isNotEmpty) {
          _selectedUser = _users.first;
        } else {
          _selectedUser = null;
        }
        _isLoadingUsers = false;
      });
    } catch (e) {
      setState(() {
        _isLoadingUsers = false;
      });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load users: $e')),
      );
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final picked = await _picker.pickImage(source: source);
      if (picked != null) {
        setState(() {
          _imageFile = File(picked.path);
          _authSuccess = null; // Clear previous result
        });
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error picking image: $e')),
      );
    }
  }

  Future<void> _verifyIdentity() async {
    if (_selectedUser == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select an enrolled user.')),
      );
      return;
    }
    if (_imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please capture or select a face image first.')),
      );
      return;
    }

    setState(() {
      _isVerifying = true;
      _authSuccess = null;
    });

    try {
      final res = await ApiService.instance.authenticateUser(
        _selectedUser!.userId,
        _imageFile!,
      );

      setState(() {
        _authSuccess = res['authenticated'] as bool? ?? false;
        _authMessage = res['status'] as String? ?? "Authentication finished.";
        _similarityScore = (res['similarity_score'] as num? ?? 0.0).toDouble();
        _livenessScore = (res['liveness_score'] as num? ?? 0.0).toDouble();
        _isVerifying = false;
      });
    } catch (e) {
      setState(() {
        _isVerifying = false;
      });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Verification error: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // User Selection Card
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Select Profile',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                  const SizedBox(height: 12),
                  if (_isLoadingUsers)
                    const Center(child: CircularProgressIndicator())
                  else if (_users.isEmpty)
                    Row(
                      children: [
                        const Icon(Icons.info_outline, color: Colors.amber),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            'No users enrolled. Use Settings to check connection or Enroll to add.',
                            style: TextStyle(color: Colors.grey[400], fontSize: 13),
                          ),
                        ),
                        IconButton(icon: const Icon(Icons.refresh), onPressed: _loadUsers),
                      ],
                    )
                  else
                    Row(
                      children: [
                        Expanded(
                          child: DropdownButtonHideUnderline(
                            child: DropdownButton<UserProfile>(
                              value: _selectedUser,
                              isExpanded: true,
                              items: _users.map((u) {
                                return DropdownMenuItem(
                                  value: u,
                                  child: Text(u.username, style: const TextStyle(fontSize: 15)),
                                );
                              }).toList(),
                              onChanged: (val) {
                                setState(() {
                                  _selectedUser = val;
                                });
                              },
                            ),
                          ),
                        ),
                        IconButton(icon: const Icon(Icons.refresh), onPressed: _loadUsers),
                      ],
                    ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Scanning Frame / Preview Area
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Container(
                    height: 250,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: const Color(0xff05060a),
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: const Color(0x14ffffff)),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(16),
                      child: Stack(
                        alignment: Alignment.center,
                        children: [
                          if (_imageFile != null)
                            Image.file(_imageFile!, fit: BoxFit.cover, width: double.infinity, height: double.infinity)
                          else
                            const Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.face_outlined, size: 64, color: Colors.grey),
                                SizedBox(height: 12),
                                Text('Capture or Select Photo', style: TextStyle(color: Colors.grey)),
                              ],
                            ),
                          
                          // Biometric HUD Overlay
                          Container(
                            margin: const EdgeInsets.all(30),
                            decoration: BoxDecoration(
                              border: Border.all(color: const Color(0xff38bdf8), width: 1.5),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Align(
                              alignment: Alignment.topCenter,
                              child: Container(
                                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                                decoration: const BoxDecoration(
                                  color: Color(0xff38bdf8),
                                  borderRadius: BorderRadius.vertical(bottom: Radius.circular(8)),
                                ),
                                child: const Text(
                                  'BIOMETRIC SCAN ZONE',
                                  style: TextStyle(fontSize: 8, fontWeight: FontWeight.bold, color: Colors.black),
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: () => _pickImage(ImageSource.camera),
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Take Photo'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0x14ffffff),
                            foregroundColor: Colors.white,
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: () => _pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Gallery'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0x14ffffff),
                            foregroundColor: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Action Button
          if (_isVerifying)
            const Center(
              child: Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 10),
                    Text('Analyzing Face Features...', style: TextStyle(color: Colors.grey)),
                  ],
                ),
              ),
            )
          else
            ElevatedButton(
              onPressed: _verifyIdentity,
              child: const Text('Verify Identity'),
            ),
          
          const SizedBox(height: 20),

          // Verification Status Banner
          if (_authSuccess != null) ...[
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: _authSuccess! ? const Color(0x1a10b981) : const Color(0x1aef4444),
                border: Border.all(
                  color: _authSuccess! ? const Color(0xff10b981) : const Color(0xffef4444),
                  width: 1,
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                children: [
                  Icon(
                    _authSuccess! ? Icons.verified_user : Icons.gpp_bad,
                    color: _authSuccess! ? const Color(0xff10b981) : const Color(0xffef4444),
                    size: 48,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _authSuccess! ? 'ACCESS GRANTED' : 'ACCESS DENIED',
                    style: TextStyle(
                      fontSize: 18, 
                      fontWeight: FontWeight.w900,
                      color: _authSuccess! ? const Color(0xff10b981) : const Color(0xffef4444),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _authMessage,
                    style: const TextStyle(fontSize: 13, color: Colors.grey),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 16),
                  
                  // Verification metrics
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      _buildMetricGauge('Similarity', _similarityScore, const Color(0xff38bdf8)),
                      _buildMetricGauge('Liveness', _livenessScore, const Color(0xff10b981)),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildMetricGauge(String label, double score, Color color) {
    return Column(
      children: [
        Stack(
          alignment: Alignment.center,
          children: [
            SizedBox(
              height: 50,
              width: 50,
              child: CircularProgressIndicator(
                value: score,
                strokeWidth: 4,
                backgroundColor: const Color(0x14ffffff),
                valueColor: AlwaysStoppedAnimation<Color>(color),
              ),
            ),
            Text(
              '${(score * 100).toInt()}%',
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12),
            ),
          ],
        ),
        const SizedBox(height: 6),
        Text(label, style: const TextStyle(fontSize: 11, color: Colors.grey)),
      ],
    );
  }
}

/// Tab 2: Enroll User
class EnrollTab extends StatefulWidget {
  const EnrollTab({super.key});

  @override
  State<EnrollTab> createState() => _EnrollTabState();
}

class _EnrollTabState extends State<EnrollTab> {
  final TextEditingController _usernameController = TextEditingController();
  final List<File> _images = [];
  final ImagePicker _picker = ImagePicker();
  bool _isRegistering = false;
  String? _successId;

  Future<void> _captureImage() async {
    if (_images.length >= 3) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Exactly 3 snaps are needed for registration.')),
      );
      return;
    }
    try {
      final picked = await _picker.pickImage(source: ImageSource.camera);
      if (picked != null) {
        setState(() {
          _images.add(File(picked.path));
        });
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error capturing photo: $e')),
      );
    }
  }

  Future<void> _submitRegistration() async {
    final username = _usernameController.text.trim();
    if (username.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a username.')),
      );
      return;
    }
    if (_images.length < 3) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Enrollment requires 3 snapshots (Selected ${_images.length}/3)')),
      );
      return;
    }

    setState(() {
      _isRegistering = true;
      _successId = null;
    });

    try {
      final res = await ApiService.instance.enrollUser(username, _images);
      setState(() {
        _isRegistering = false;
        _successId = res['user_id']?.toString();
        _images.clear();
        _usernameController.clear();
      });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('User registered successfully!')),
      );
    } catch (e) {
      setState(() {
        _isRegistering = false;
      });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Registration failed: $e')),
      );
    }
  }

  @override
  void dispose() {
    _usernameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Registration Form
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Enroll New User',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _usernameController,
                    decoration: const InputDecoration(
                      labelText: 'Username',
                      hintText: 'e.g. Alice',
                      prefixIcon: Icon(Icons.person),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Snapshots acquired list
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'Acquisition Snapshots',
                        style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
                      ),
                      Text(
                        '${_images.length} / 3 snaps',
                        style: TextStyle(
                          color: _images.length == 3 ? const Color(0xff10b981) : Colors.grey,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  
                  if (_images.isEmpty)
                    Container(
                      height: 120,
                      alignment: Alignment.center,
                      decoration: BoxDecoration(
                        color: const Color(0xff05060a),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.add_a_photo_outlined, color: Colors.grey, size: 36),
                          SizedBox(height: 6),
                          Text('No photos captured yet', style: TextStyle(color: Colors.grey, fontSize: 13)),
                        ],
                      ),
                    )
                  else
                    SizedBox(
                      height: 120,
                      child: ListView.builder(
                        scrollDirection: Axis.horizontal,
                        itemCount: _images.length,
                        itemBuilder: (context, index) {
                          return Stack(
                            children: [
                              Container(
                                width: 100,
                                margin: const EdgeInsets.only(right: 12),
                                decoration: BoxDecoration(
                                  borderRadius: BorderRadius.circular(12),
                                  border: Border.all(color: const Color(0x14ffffff)),
                                  image: DecorationImage(
                                    image: FileImage(_images[index]),
                                    fit: BoxFit.cover,
                                  ),
                                ),
                              ),
                              Positioned(
                                top: 0,
                                right: 12,
                                child: InkWell(
                                  onTap: () {
                                    setState(() {
                                      _images.removeAt(index);
                                    });
                                  },
                                  child: Container(
                                    decoration: const BoxDecoration(
                                      color: Color(0xffef4444),
                                      shape: BoxShape.circle,
                                    ),
                                    padding: const EdgeInsets.all(4),
                                    child: const Icon(Icons.close, size: 14, color: Colors.white),
                                  ),
                                ),
                              ),
                            ],
                          );
                        },
                      ),
                    ),
                  
                  const SizedBox(height: 16),
                  ElevatedButton.icon(
                    onPressed: _images.length >= 3 ? null : _captureImage,
                    icon: const Icon(Icons.add_a_photo),
                    label: const Text('Capture Snapshot'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0x14ffffff),
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 20),

          if (_isRegistering)
            const Center(
              child: Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 10),
                  Text('Uploading Face Datasets & Training Centroids...', style: TextStyle(color: Colors.grey)),
                ],
              ),
            )
          else
            ElevatedButton(
              onPressed: _images.length == 3 ? _submitRegistration : null,
              child: const Text('Register Biometrics'),
            ),

          if (_successId != null) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0x1a10b981),
                border: Border.all(color: const Color(0xff10b981)),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  const Icon(Icons.check_circle_outline, color: Color(0xff10b981)),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Enrollment Successful!',
                          style: TextStyle(fontWeight: FontWeight.bold, color: Color(0xff10b981)),
                        ),
                        Text(
                          'Enrolled User ID: $_successId',
                          style: const TextStyle(fontSize: 10, color: Colors.grey),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/// Tab 3: Analytics Logs & Metrics
class AnalyticsTab extends StatefulWidget {
  const AnalyticsTab({super.key});

  @override
  State<AnalyticsTab> createState() => _AnalyticsTabState();
}

class _AnalyticsTabState extends State<AnalyticsTab> {
  bool _isLoading = false;
  Map<String, dynamic> _metrics = {};
  List<LogEntry> _logs = [];

  @override
  void initState() {
    super.initState();
    _refreshData();
  }

  Future<void> _refreshData() async {
    setState(() {
      _isLoading = true;
    });

    try {
      final metricsData = await ApiService.instance.fetchMetrics();
      final logsData = await ApiService.instance.fetchLogs();
      setState(() {
        _metrics = metricsData;
        _logs = logsData;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load metrics: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    return RefreshIndicator(
      onRefresh: _refreshData,
      child: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          // Stats Header Grid
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Performance Stats',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
              IconButton(icon: const Icon(Icons.refresh), onPressed: _refreshData),
            ],
          ),
          const SizedBox(height: 12),
          GridView.count(
            crossAxisCount: 2,
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            crossAxisSpacing: 12,
            mainAxisSpacing: 12,
            childAspectRatio: 1.5,
            children: [
              _buildMetricCard(
                'Equal Error Rate',
                '${((_metrics['eer'] as num? ?? 0.0) * 100).toStringAsFixed(2)}%',
                Icons.error_outline,
              ),
              _buildMetricCard(
                'ROC Area (AUC)',
                (_metrics['auc'] as num? ?? 0.0).toStringAsFixed(4),
                Icons.analytics_outlined,
              ),
              _buildMetricCard(
                'System Accuracy',
                '${((_metrics['accuracy'] as num? ?? 0.0) * 100).toStringAsFixed(1)}%',
                Icons.check_circle_outline,
              ),
              _buildMetricCard(
                'Audit Trials',
                '${_metrics['total_attempts'] ?? 0}',
                Icons.history,
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Logs List Section
          const Text(
            'Chronological Audit Logs',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),
          
          if (_logs.isEmpty)
            Container(
              height: 150,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: const Color(0xff161c2d).withOpacity(0.3),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0x14ffffff)),
              ),
              child: const Text('No login attempts recorded yet.', style: TextStyle(color: Colors.grey)),
            )
          else
            ListView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: _logs.length,
              itemBuilder: (context, index) {
                final log = _logs[index];
                return Card(
                  margin: const EdgeInsets.only(bottom: 10),
                  child: ListTile(
                    leading: CircleAvatar(
                      backgroundColor: log.authenticated ? const Color(0x1a10b981) : const Color(0x1aef4444),
                      child: Icon(
                        log.authenticated ? Icons.check : Icons.close,
                        color: log.authenticated ? const Color(0xff10b981) : const Color(0xffef4444),
                      ),
                    ),
                    title: Text(log.username, style: const TextStyle(fontWeight: FontWeight.bold)),
                    subtitle: Text(
                      'Match: ${(log.similarityScore * 100).toStringAsFixed(0)}%  |  Liveness: ${(log.livenessScore * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(fontSize: 12, color: Colors.grey),
                    ),
                    trailing: Text(
                      _formatDate(log.timestamp),
                      style: const TextStyle(fontSize: 10, color: Colors.grey),
                    ),
                  ),
                );
              },
            ),
        ],
      ),
    );
  }

  String _formatDate(String isoString) {
    try {
      final dateTime = DateTime.parse(isoString);
      return DateFormat('HH:mm  |  MM/dd').format(dateTime);
    } catch (_) {
      return isoString;
    }
  }

  Widget _buildMetricCard(String label, String value, IconData icon) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Icon(icon, color: const Color(0xff38bdf8), size: 20),
                const SizedBox(),
              ],
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  value,
                  style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w900, color: Colors.white),
                ),
                Text(
                  label,
                  style: const TextStyle(fontSize: 10, color: Colors.grey),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// Tab 4: Settings
class SettingsTab extends StatefulWidget {
  const SettingsTab({super.key});

  @override
  State<SettingsTab> createState() => _SettingsTabState();
}

class _SettingsTabState extends State<SettingsTab> {
  final TextEditingController _urlController = TextEditingController();
  double _matchThreshold = 0.80;
  double _livenessThreshold = 0.85;
  bool _testingConnection = false;
  bool? _isConnected;

  @override
  void initState() {
    super.initState();
    _urlController.text = AppConfig.instance.baseUrl;
    _matchThreshold = AppConfig.instance.matchThreshold;
    _livenessThreshold = AppConfig.instance.livenessThreshold;
  }

  Future<void> _testConnection() async {
    setState(() {
      _testingConnection = true;
      _isConnected = null;
    });
    
    // Temporarily save base URL to run checkHealth() on it
    final oldUrl = AppConfig.instance.baseUrl;
    final newUrl = _urlController.text.trim();
    await AppConfig.instance.setBaseUrl(newUrl);

    try {
      final active = await ApiService.instance.checkHealth();
      setState(() {
        _testingConnection = false;
        _isConnected = active;
      });
      if (!active) {
        // Restore old url if new one failed health check
        await AppConfig.instance.setBaseUrl(oldUrl);
      }
    } catch (_) {
      setState(() {
        _testingConnection = false;
        _isConnected = false;
      });
      // Restore old url
      await AppConfig.instance.setBaseUrl(oldUrl);
    }
  }

  Future<void> _saveSettings() async {
    final url = _urlController.text.trim();
    if (url.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('URL cannot be empty.')),
      );
      return;
    }

    try {
      await AppConfig.instance.setBaseUrl(url);
      await AppConfig.instance.setMatchThreshold(_matchThreshold);
      await AppConfig.instance.setLivenessThreshold(_livenessThreshold);

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Settings saved successfully!')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to save settings: $e')),
      );
    }
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Connection Card
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Server Configuration',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _urlController,
                    decoration: const InputDecoration(
                      labelText: 'Server Base URL',
                      hintText: 'e.g. https://my-server.onrender.com',
                      prefixIcon: Icon(Icons.link),
                    ),
                  ),
                  const SizedBox(height: 12),
                  
                  if (_testingConnection)
                    const Center(child: CircularProgressIndicator())
                  else
                    ElevatedButton.icon(
                      onPressed: _testConnection,
                      icon: const Icon(Icons.network_ping),
                      label: const Text('Test Connection'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0x14ffffff),
                        foregroundColor: Colors.white,
                      ),
                    ),
                  
                  if (_isConnected != null) ...[
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Icon(
                          _isConnected! ? Icons.check_circle : Icons.error,
                          color: _isConnected! ? const Color(0xff10b981) : const Color(0xffef4444),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _isConnected! ? 'Connected to Backend Successfully!' : 'Failed to connect to backend.',
                          style: TextStyle(
                            color: _isConnected! ? const Color(0xff10b981) : const Color(0xffef4444),
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Thresholds Card
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Biometric Parameters',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 16),
                  
                  // Match Threshold Slider
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Similarity Match Threshold', style: TextStyle(fontSize: 13)),
                      Text('${(_matchThreshold * 100).toInt()}%', style: const TextStyle(fontWeight: FontWeight.bold)),
                    ],
                  ),
                  Slider(
                    value: _matchThreshold,
                    min: 0.1,
                    max: 0.99,
                    onChanged: (val) {
                      setState(() {
                        _matchThreshold = val;
                      });
                    },
                  ),
                  
                  const SizedBox(height: 12),

                  // Liveness Threshold Slider
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Liveness spoofing Threshold', style: TextStyle(fontSize: 13)),
                      Text('${(_livenessThreshold * 100).toInt()}%', style: const TextStyle(fontWeight: FontWeight.bold)),
                    ],
                  ),
                  Slider(
                    value: _livenessThreshold,
                    min: 0.1,
                    max: 0.99,
                    onChanged: (val) {
                      setState(() {
                        _livenessThreshold = val;
                      });
                    },
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),

          ElevatedButton(
            onPressed: _saveSettings,
            child: const Text('Save Configuration'),
          ),
        ],
      ),
    );
  }
}
