#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Modulino.h>

#define SDA_PIN 8
#define SCL_PIN 9

const char* WIFI_SSID = "SiddhuiPhone";
const char* WIFI_PASS = "#siddhu123";

IPAddress local_IP(172, 20, 10, 14);
IPAddress gateway(172, 20, 10, 1);
IPAddress subnet(255, 255, 255, 240);

ModulinoMovement movement;
ModulinoVibro vibro;
WebServer server(80);

String cmd = "";

// Status flags
bool movementOk = false;
bool vibroOk = false;
bool wifiOk = false;
bool serverStarted = false;
bool vibrationEnabled = false;

// Latest sensor values
float ax = 0.0f, ay = 0.0f, az = 0.0f;
float gx = 0.0f, gy = 0.0f, gz = 0.0f;
unsigned long lastReadMs = 0;

// Non-blocking vibration pattern state
unsigned long lastWarnMs = 0;
int warnStep = 0;

// -------------------- SAFE STARTUP WINDOW --------------------
void safeStartupWindow() {
  Serial.println("Safe startup window: 5 seconds");
  Serial.println("Reflash now if needed...");
  unsigned long start = millis();
  while (millis() - start < 5000) {
    delay(10);
  }
}

// -------------------- VIBRATION --------------------
void vibroOffSafe() {
  if (vibroOk) {
    vibro.off();
  }
}

void vibroSingleBuzz(int ms = 200, int power = 220) {
  if (!vibroOk) return;
  vibro.on(ms, true, power);
  vibro.off();
}

void triggerWarningBuzzOnce() {
  if (!vibroOk) return;

  vibro.on(80, true, 220);
  delay(120);

  vibro.on(80, true, 220);
  delay(120);

  vibro.on(80, true, 220);
  delay(120);

  vibro.off();
}

void handleWarningBuzzNonBlocking() {
  if (!vibroOk || !vibrationEnabled) return;

  unsigned long now = millis();

  switch (warnStep) {
    case 0:
      vibro.on(80, true, 220);
      lastWarnMs = now;
      warnStep = 1;
      break;

    case 1:
      if (now - lastWarnMs >= 120) {
        vibro.on(80, true, 220);
        lastWarnMs = now;
        warnStep = 2;
      }
      break;

    case 2:
      if (now - lastWarnMs >= 120) {
        vibro.on(80, true, 220);
        lastWarnMs = now;
        warnStep = 3;
      }
      break;

    case 3:
      if (now - lastWarnMs >= 120) {
        vibro.off();
        lastWarnMs = now;
        warnStep = 4;
      }
      break;

    case 4:
      if (now - lastWarnMs >= 200) {
        warnStep = 0;
      }
      break;
  }
}

// -------------------- MOVEMENT --------------------
void updateMovement() {
  if (!movementOk) return;

  if (movement.update()) {
    ax = movement.getX();
    ay = movement.getY();
    az = movement.getZ();

    gx = movement.getRoll();
    gy = movement.getPitch();
    gz = movement.getYaw();

    lastReadMs = millis();
  }
}

void printMovement() {
  updateMovement();

  Serial.print("ACC: ");
  Serial.print(ax, 4);
  Serial.print(", ");
  Serial.print(ay, 4);
  Serial.print(", ");
  Serial.print(az, 4);

  Serial.print(" | GYRO: ");
  Serial.print(gx, 4);
  Serial.print(", ");
  Serial.print(gy, 4);
  Serial.print(", ");
  Serial.println(gz, 4);
}

// -------------------- SERIAL COMMANDS --------------------
void handleSerialCommands() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      cmd.trim();
      cmd.toLowerCase();

      if (cmd == "on") {
        vibrationEnabled = true;
        warnStep = 0;
        Serial.println("Vibration enabled");
      }
      else if (cmd == "off") {
        vibrationEnabled = false;
        warnStep = 0;
        vibroOffSafe();
        Serial.println("Vibration disabled");
      }
      else if (cmd == "warn") {
        Serial.println("Warning buzz");
        triggerWarningBuzzOnce();
      }
      else if (cmd == "buzz") {
        Serial.println("Single buzz");
        vibroSingleBuzz(200, 220);
      }
      else if (cmd == "status") {
        printMovement();
      }
      else if (cmd == "wifi") {
        Serial.print("WiFi.status() = ");
        Serial.println((int)WiFi.status());
        Serial.print("SSID = ");
        Serial.println(WiFi.SSID());
        Serial.print("IP = ");
        Serial.println(WiFi.localIP());
      }
      else if (cmd.length() > 0) {
        Serial.print("Unknown command: ");
        Serial.println(cmd);
      }

      cmd = "";
    } else {
      cmd += c;
    }
  }
}

// -------------------- WEB PAGE --------------------
String htmlPage() {
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32 Movement + Vibro</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background:#0f1115; color:#f5f7fa; }
    .card { background:#1a1f29; padding:20px; border-radius:16px; margin-bottom:18px; }
    .row { margin: 8px 0; font-size: 18px; }
    button {
      font-size: 18px; padding: 12px 18px; margin: 6px;
      border: none; border-radius: 12px; cursor: pointer;
    }
    .on { background:#22c55e; color:white; }
    .off { background:#ef4444; color:white; }
    .warn { background:#f59e0b; color:white; }
    .buzz { background:#3b82f6; color:white; }
    h1, h2 { margin-top: 0; }
    .small { color:#aab3c2; font-size:14px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>ESP32 Movement + Vibro</h1>
    <div class="row">Vibration enabled: <span id="vibenabled">--</span></div>
    <div class="row">WiFi connected: <span id="wifiok">--</span></div>
    <div class="row">SSID: <span id="ssid">--</span></div>
    <div class="row">IP: <span id="ip">--</span></div>
    <div class="row small">This page refreshes sensor data automatically.</div>
  </div>

  <div class="card">
    <h2>Controls</h2>
    <button class="on" onclick="sendCmd('/on')">Enable Vibration</button>
    <button class="off" onclick="sendCmd('/off')">Disable Vibration</button>
    <button class="buzz" onclick="sendCmd('/buzz')">Single Buzz</button>
    <button class="warn" onclick="sendCmd('/warn')">Warning Buzz</button>
  </div>

  <div class="card">
    <h2>Movement Data</h2>
    <div class="row">ACC X: <span id="ax">--</span></div>
    <div class="row">ACC Y: <span id="ay">--</span></div>
    <div class="row">ACC Z: <span id="az">--</span></div>
    <div class="row">GYRO Roll: <span id="gx">--</span></div>
    <div class="row">GYRO Pitch: <span id="gy">--</span></div>
    <div class="row">GYRO Yaw: <span id="gz">--</span></div>
    <div class="row small">Last update ms: <span id="last">--</span></div>
    <div class="row small">Movement OK: <span id="movementOk">--</span></div>
    <div class="row small">Vibro OK: <span id="vibroOk">--</span></div>
  </div>

  <script>
    async function fetchData() {
      try {
        const res = await fetch('/data');
        const data = await res.json();
        document.getElementById('ax').textContent = Number(data.ax).toFixed(4);
        document.getElementById('ay').textContent = Number(data.ay).toFixed(4);
        document.getElementById('az').textContent = Number(data.az).toFixed(4);
        document.getElementById('gx').textContent = Number(data.gx).toFixed(4);
        document.getElementById('gy').textContent = Number(data.gy).toFixed(4);
        document.getElementById('gz').textContent = Number(data.gz).toFixed(4);
        document.getElementById('last').textContent = data.lastReadMs;
        document.getElementById('vibenabled').textContent = data.vibrationEnabled ? 'YES' : 'NO';
        document.getElementById('wifiok').textContent = data.wifiOk ? 'YES' : 'NO';
        document.getElementById('ssid').textContent = data.ssid;
        document.getElementById('ip').textContent = data.ip;
        document.getElementById('movementOk').textContent = data.movementOk ? 'YES' : 'NO';
        document.getElementById('vibroOk').textContent = data.vibroOk ? 'YES' : 'NO';
      } catch (e) {
        console.log('Fetch failed');
      }
    }

    async function sendCmd(path) {
      try {
        await fetch(path);
        setTimeout(fetchData, 150);
      } catch (e) {
        console.log('Command failed');
      }
    }

    fetchData();
    setInterval(fetchData, 400);
  </script>
</body>
</html>
)rawliteral";
  return html;
}

// -------------------- WEB HANDLERS --------------------
void handleRoot() {
  server.send(200, "text/html", htmlPage());
}

void handleData() {
  updateMovement();

  String json = "{";
  json += "\"ax\":" + String(ax, 6) + ",";
  json += "\"ay\":" + String(ay, 6) + ",";
  json += "\"az\":" + String(az, 6) + ",";
  json += "\"gx\":" + String(gx, 6) + ",";
  json += "\"gy\":" + String(gy, 6) + ",";
  json += "\"gz\":" + String(gz, 6) + ",";
  json += "\"lastReadMs\":" + String(lastReadMs) + ",";
  json += "\"vibrationEnabled\":" + String(vibrationEnabled ? "true" : "false") + ",";
  json += "\"movementOk\":" + String(movementOk ? "true" : "false") + ",";
  json += "\"vibroOk\":" + String(vibroOk ? "true" : "false") + ",";
  json += "\"wifiOk\":" + String(wifiOk ? "true" : "false") + ",";
  json += "\"ssid\":\"" + String(wifiOk ? WiFi.SSID() : "") + "\",";
  json += "\"ip\":\"" + String(wifiOk ? WiFi.localIP().toString() : "") + "\"";
  json += "}";

  server.send(200, "application/json", json);
}

void handleWiFiStatus() {
  String json = "{";
  json += "\"wifiConnected\":" + String(wifiOk ? "true" : "false") + ",";
  json += "\"ssid\":\"" + String(wifiOk ? WiFi.SSID() : "") + "\",";
  json += "\"ip\":\"" + String(wifiOk ? WiFi.localIP().toString() : "") + "\"";
  json += "}";

  server.send(200, "application/json", json);
}

void handleOn() {
  vibrationEnabled = true;
  warnStep = 0;
  server.send(200, "text/plain", "Vibration enabled");
}

void handleOff() {
  vibrationEnabled = false;
  warnStep = 0;
  vibroOffSafe();
  server.send(200, "text/plain", "Vibration disabled");
}

void handleBuzz() {
  vibroSingleBuzz(200, 220);
  server.send(200, "text/plain", "Single buzz");
}

void handleWarn() {
  triggerWarningBuzzOnce();
  server.send(200, "text/plain", "Warning buzz");
}

// -------------------- WIFI --------------------
void setupWiFi() {
  WiFi.mode(WIFI_STA);

  // Set static IP
  if (!WiFi.config(local_IP, gateway, subnet)) {
    Serial.println("Static IP failed");
  }

  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting to WiFi");

  unsigned long startAttempt = millis();
  const unsigned long timeoutMs = 15000;

  while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < timeoutMs) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();

  wifiOk = (WiFi.status() == WL_CONNECTED);

  if (wifiOk) {
    Serial.println("WIFI_CONNECTED");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());  // should ALWAYS be 172.20.10.3
  } else {
    Serial.println("WIFI_FAILED");
  }
}
// -------------------- SETUP --------------------
void setup() {
  Serial.begin(115200);
  delay(500);

  safeStartupWindow();

  Serial.println("Starting Modulino Movement + Vibro + WiFi");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Modulino.begin();

  movementOk = movement.begin();
  if (!movementOk) {
    Serial.println("Movement module NOT found");
  } else {
    Serial.println("Movement module detected");
  }

  vibroOk = vibro.begin();
  if (!vibroOk) {
    Serial.println("Vibro module NOT found");
  } else {
    Serial.println("Vibro module detected");
    vibroSingleBuzz(150, 200);
  }

  setupWiFi();

  if (wifiOk) {
    server.on("/", handleRoot);
    server.on("/data", handleData);
    server.on("/wifi", handleWiFiStatus);
    server.on("/on", handleOn);
    server.on("/off", handleOff);
    server.on("/buzz", handleBuzz);
    server.on("/warn", handleWarn);
    server.begin();
    serverStarted = true;

    Serial.println("Web server started");
    Serial.println("Open this on your phone/computer:");
    Serial.print("http://");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("Web server not started");
  }

  Serial.println("Serial commands:");
  Serial.println("  on");
  Serial.println("  off");
  Serial.println("  buzz");
  Serial.println("  warn");
  Serial.println("  status");
  Serial.println("  wifi");
  Serial.println("Setup complete");
}

// -------------------- LOOP --------------------
void loop() {
  if (serverStarted && wifiOk) {
    server.handleClient();
  }

  handleSerialCommands();
  updateMovement();
  handleWarningBuzzNonBlocking();

  delay(10);
}