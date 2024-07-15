import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_sms_inbox/flutter_sms_inbox.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:http/http.dart' as http;
import 'package:sms_flutter/models/smsmodel.dart';
import 'package:sms_flutter/pages/SplashScreen.dart';
import 'package:loading_indicator/loading_indicator.dart';


class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  SmsQuery query = SmsQuery();
  List<SmsMessage> messages = [];
  List<SmsModel>finalList=[];
  bool showSplash=true;
  Future<void>getSms()async{
    messages=await query.getAllSms;
    int length=0;
    if(messages.length<100){
      length=messages.length;
    }else{
      length=100;
    }
    for(int i=0;i<length;i++) {
      print(i);
      final url = Uri.parse("https://spammsg.onrender.com/predictMobile");
      final response = await http.post(url, body: jsonEncode(
          {"message": messages[i].body??''}),
          headers: {'Content-type': 'application/json'});
      SmsModel s=SmsModel(sms: messages[i], result: jsonDecode(response.body)['result']);
      finalList.add(s);
    }
    setState(() {
      showSplash=false;
      finalList=finalList;
    });
  }
  @override
  void initState() {
    super.initState();
    getSms();
  }
  @override
  Widget build(BuildContext context) {
    return showSplash==false?Scaffold(
      appBar:AppBar(
        backgroundColor: Colors.blueAccent,
        centerTitle: true,
        title: Text("SMS Spam Detection",style: TextStyle(color:Colors.white),),
      ),
      body: ListView.builder(itemBuilder: (context,index){
        return Card(
          color: finalList[index].result == "Not Spam" ? Colors.green[100] : Colors.redAccent[100],
          child: ListTile(
            leading: finalList[index].result == "Not Spam" ? Icon(Icons.safety_check_outlined) : Icon(Icons.dangerous_sharp),
            minLeadingWidth: 4,
            title: Text(finalList[index].sms.body ?? 'empty'),
            subtitle: Text(finalList[index].sms.address ?? 'empty'),
          ),
        );
      },itemCount: finalList.length,),
      floatingActionButton: FloatingActionButton(
        onPressed: (){
          Navigator.pushNamed(context,'/custom');
        },
        backgroundColor: Colors.blueAccent,
        child: Icon(Icons.dashboard_customize,color: Colors.white,),

      ),
    ):Container(
      color: Colors.blueAccent,
      padding: EdgeInsets.all(20.0), // Add padding for spacing
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center, // Center the content vertically
          children: [
            Center(
              child: SizedBox(
                height: 70,
                child: LoadingIndicator(
                  indicatorType: Indicator.lineScaleParty,
                  colors: [Colors.white, Colors.amber],
                  strokeWidth: 4.0,
                ),
              ),
            ), // Add space between the loading indicator and text
            SizedBox(height: 20,),
            Text(
              "Loading Messages",
              style: TextStyle(
                color: Colors.white,
                fontSize: 30,
                fontWeight: FontWeight.bold,
                decoration: TextDecoration.none, // Remove underline
              ),
            ),
          ],
        ),
      ),
    );



  }
}
