import 'package:animated_splash_screen/animated_splash_screen.dart';
import 'package:flutter/material.dart';
import 'package:sms_flutter/pages/HomeScreen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({Key? key}) : super(key: key);

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  Widget build(BuildContext context) {
    return AnimatedSplashScreen(
        splash:Icon(Icons.message,color: Colors.white,size: 50,),
        duration: 1000,
        nextScreen: const HomeScreen(),
        backgroundColor: Colors.blueAccent,
    );
  }
}
