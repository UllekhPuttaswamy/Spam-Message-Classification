import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class CustomCheck extends StatefulWidget {
  const CustomCheck({Key? key}) : super(key: key);

  @override
  State<CustomCheck> createState() => _CustomCheckState();
}

class _CustomCheckState extends State<CustomCheck> {
  TextEditingController t1 = TextEditingController();
  String result = '';

  Future<void> _checkSpam() async {
    final url = Uri.parse("https://spammsg.onrender.com/predictMobile");
    final response = await http.post(
      url,
      body: jsonEncode({"message": t1.text}),
      headers: {'Content-type': 'application/json'},
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      setState(() {
        result = data['result'];
      });
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            backgroundColor: Colors.lightBlueAccent[50],
            title: Text('Result from our model'),
            content: Text(result),
            actions: <Widget>[
              TextButton(
                onPressed: () {
                  Navigator.of(context).pop();
                },
                child: Text('Close'),
              ),
            ],
          );
        },
      );
    } else {
      throw Exception('Failed to load data');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.blueAccent,
        title: Text(
          "Enter Your Custom Message",
          style: TextStyle(color: Colors.white),
        ),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "Enter Message",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            TextField(
              controller: t1,
              maxLines: 6,
              decoration: InputDecoration(
                hintText: "Type your message here",
                border: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.blueAccent), // Change border color
                ),
              ),
            ),
            SizedBox(height: 20),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _checkSpam,
                style: ButtonStyle(
                  backgroundColor: MaterialStateColor.resolveWith((states) => Colors.blueAccent), // Set button color
                ),
                child: Text(
                  "Check if spam or not",
                  style: TextStyle(color: Colors.white), // Set text color
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

