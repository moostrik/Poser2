// ==========================================
// CONFIGURATIE
// ==========================================
//#define TEST_MODE // Uncomment voor testmodus
//nog doen. Wanneer RPM==0 de 4 DACS data halen uit White[0],White[1800] en blu... etc

#define ETHERNET_LARGE_BUFFERS 1
#define MAX_SOCK_NUM 4
int ISERNOGETHERNET=10;

unsigned long vmillis;

unsigned int TEST=900;
unsigned int testteller=0;
//unsigned int RPM=0;
unsigned int vRPM=0;
//bool SLOW=true;
//bool NULPUNT=false;
// Volatile variabelen voor veilige communicatie tussen Core 0 en Core 1
volatile unsigned int RPM = 0;
volatile bool SLOW = true;
volatile bool NULPUNT = false;

// Fase-correcties die live via UDP kunnen worden aangepast
volatile int cor0 = 0; // wit 1 bovenste witte
volatile int cor2 = 3; // wit 2
volatile int cor1 = 1; // blauw1 bovenste blauwe
volatile int cor3 = 0; // blauw2


#include <Arduino.h>
#include <Ethernet_Generic.h>
#include <SPI.h>
static int Wteller = 0;
// ---- PINOUT ----
#define W5500_CS   17
#define W5500_RST  20
#ifndef LED_BUILTIN
#define LED_BUILTIN 25
#endif

// Nieuwe DAC Pinnen (SPI1 hardware pinnen)
#define DAC_SPI      SPI1
static const uint8_t PIN_SCK     = 10;
static const uint8_t PIN_MOSI    = 11;
static const uint8_t PIN_CS_DAC  = 13; 
static const uint8_t PIN_SENSOR  = 15; 

#include <Arduino.h>
#include <SPI.h>
#include "hardware/spi.h"
#include "hardware/address_mapped.h"

//cor positief = naar links vershuiven
// int cor0=0; //wit 1 bovenste witte
// int cor2=3; //wit 2
// int cor1=1; //blauw1 bovenste blauwe
// int cor3=0; //blauw2

// 3. SNELLERE MACROS
static inline void sioSet(uint pin) { sio_hw->gpio_set = 1u << pin; }
static inline void sioClr(uint pin) { sio_hw->gpio_clr = 1u << pin; }
#define SPI_HW spi1



// 5. DAC SCHRIJF-FUNCTIE
static inline void ad7304_write_turbo(uint8_t channel, uint8_t data) {
    sioClr(PIN_CS_DAC);
    
    // We duwen beide bytes direct achter elkaar in de FIFO buffer
    while (!spi_is_writable(SPI_HW));
    spi_get_hw(SPI_HW)->dr = (uint32_t)(0x0C | channel); // Byte 1
    
    while (!spi_is_writable(SPI_HW));
    spi_get_hw(SPI_HW)->dr = (uint32_t)data;             // Byte 2

    // Wacht tot de FIFO helemaal leeg is en de laatste bit verzonden
    while (spi_is_busy(SPI_HW));
    
    sioSet(PIN_CS_DAC);
}

// 6. DE REST VAN JE CORE 1 CODE (setup1 en loop1)...


// Netwerk
static byte mac[] = {0xDE,0xAD,0xBE,0xEF,0x55,0x00};
static IPAddress ip (192,168,1,177);
static IPAddress gw (192,168,1,1);
static IPAddress sn (255,255,255,0);
static const uint16_t localPort = 6454;
EthernetUDP Udp;
static unsigned char Data[6][1200];
static uint8_t WIT[3600];
static uint8_t BLAUW[3600];
static uint8_t recvMask = 0;
static uint32_t rx_total = 0, drop_total = 0;

IPAddress artnetMasterIP(192,168,1,123); // fallback, change if useful
uint16_t artnetMasterPort = 6454;
bool haveArtnetMaster = false;

// ==========================================
// CORE 1: ALLES-IN-EEN (DAC + SENSOR)
// ==========================================

// Variabelen voor PLL (Alles float voor snelheid)
#define MIN_INTERVAL 500
#define MAX_INTERVAL 1000000
#define ALPHA 0.95f // 'f' forceert float

float pll_interval = 30000.0f;
float step = 8.33f; // Startwaarde
uint64_t start_time = 0;
bool vsensor = HIGH; 
float loopL = 3600.0f;

// Functie om de snelheid dynamisch te berekenen en te versturen
void setSpeed(uint16_t snelheid) {
  uint8_t msg[8];
  
  msg[0] = 0x01;                       // Slave ID
  msg[1] = 0x06;                       // Function Code
  msg[2] = 0x81;                       // Register High
  msg[3] = 0x92;                       // Register Low
  msg[4] = (snelheid >> 8) & 0xFF;     // Snelheid High byte
  msg[5] = snelheid & 0xFF;            // Snelheid Low byte
  
  // Bereken de CRC-16 checksum voor de eerste 6 bytes
  uint16_t crc = crc16_modbus(msg, 6);
  
  // Modbus CRC is Little Endian (Low byte eerst)
  msg[6] = crc & 0xFF;
  msg[7] = (crc >> 8) & 0xFF;
  
  // Verstuur het 8-bytes pakket
  Serial1.write(msg, 8);
  Serial1.flush(); 
}

// De Modbus CRC-16 berekening
uint16_t crc16_modbus(uint8_t *data, uint16_t len) {
  uint16_t crc = 0xFFFF;
  for (uint16_t i = 0; i < len; i++) {
    crc ^= (uint16_t)data[i];
    for (int j = 8; j != 0; j--) {
      if ((crc & 0x0001) != 0) {
        crc >>= 1;
        crc ^= 0xA001;
      } else {
        crc >>= 1;
      }
    }
  }
  return crc;
}
// Functie om de motor AAN te zetten (01 06 07 D0 00 00 + CRC 89 47)
void motorOn() {
  uint8_t cmd[] = {0x01, 0x06, 0x07, 0xD0, 0x00, 0x00, 0x89, 0x47};
  Serial1.write(cmd, 8);
  Serial1.flush();
  Serial.println(" -> Commando verzonden: Motor ON");
}

// Functie om de motor UIT te zetten (01 06 07 D0 00 01 + CRC 48 87)
void motorOff() {
  uint8_t cmd[] = {0x01, 0x06, 0x07, 0xD0, 0x00, 0x01, 0x48, 0x87};
  Serial1.write(cmd, 8);
  Serial1.flush();
  Serial.println(" -> Commando verzonden: Motor OFF");
}

void setup1() {
  pinMode(PIN_SENSOR, INPUT_PULLUP); // PULLUP niet nodig als sensor actief pushed, anders INPUT_PULLUP
  pinMode(PIN_CS_DAC, OUTPUT);    
  digitalWrite(PIN_CS_DAC, HIGH);

  DAC_SPI.setSCK(PIN_SCK);
  DAC_SPI.setTX(PIN_MOSI);
  DAC_SPI.begin();
  // 20 MHz is veilig
  DAC_SPI.beginTransaction(SPISettings(48000000, MSBFIRST, SPI_MODE0)); 
}

// Extra variabele bovenaan je code
float phase_error_accumulator = 0.0f; 

void loop1() {
  static double next_t_us = 0;
  static int Wteller = 0;

  // 1. SENSOR CHECK met Phase-correction
  bool sensor = !gpio_get(PIN_SENSOR); 
  gpio_put(LED_BUILTIN, sensor);


  {

    if (sensor == LOW && vsensor == HIGH) { 

      ISERNOGETHERNET++;

      if(RPM<200)NULPUNT=true;
      else NULPUNT=false;
      uint64_t now = time_us_64();
      if (start_time != 0) {
        float delta = (float)(now - start_time);
        if (delta > MIN_INTERVAL && delta < MAX_INTERVAL) {
          // Standaard PLL interval (frequentie volgen)
          pll_interval = ALPHA * delta + (1.0f - ALPHA) * pll_interval;
          
          // --- DE MAGIE: Phase Error ---
          // Hoeveel samples zaten we ernaast? (3600 is het doel)
          float error = 3600.0f - (float)Wteller; 
          
          // Pas de stapgrootte aan: als error > 0 (te laat), maak stapjes groter.
          // We corrigeren een klein deel van de fout (bijv. 10%) per ronde
          float correction = (error * 0.10f); 
          step = (pll_interval / (3600.0f + correction));
        }
      }
      start_time = now;
      
      // Harde Sync blijft nodig om jitter te voorkomen
      next_t_us = (double)now; 
      Wteller = 0;
    }
    vsensor = sensor;

    // 2. DAC OUTPUT TIMER (Immuun voor schrijfduur)
    double now_d = (double)time_us_64();
    
    if(SLOW==false)
    {

      if (now_d >= next_t_us) {
        next_t_us += (double)step; 

        if (Wteller < 3600) {
          ad7304_write_turbo(0, WIT[(Wteller + 0+TEST+cor0) % 3600]); 
          ad7304_write_turbo(1, BLAUW[(Wteller + 1800+900+TEST+cor1) % 3600]);
          ad7304_write_turbo(2,WIT[(Wteller + 1800 + TEST+cor2) % 3600]); 
          ad7304_write_turbo(3, BLAUW[(Wteller + 900+TEST+cor3) % 3600]);
    /*
          ad7304_write_turbo(0, 255); 
          ad7304_write_turbo(1, 255);
          ad7304_write_turbo(2,255); 
          ad7304_write_turbo(3, 255);
    */


          Wteller++; 
        }
      }
      
    }
    
    else      
      {
          ad7304_write_turbo(0, WIT[0]); 
          ad7304_write_turbo(1, BLAUW[0]);
          ad7304_write_turbo(2,WIT[1800]); 
          ad7304_write_turbo(3, BLAUW[1800]);
      }
  }
}
// ==========================================
// CORE 0: ALLEEN UDP
// ==========================================
void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  // Boot knipper
  for (int i=0;i<3;i++){ digitalWrite(LED_BUILTIN, HIGH); delay(80); digitalWrite(LED_BUILTIN, LOW); delay(80); }

  Serial.begin(115200);
  Serial1.begin(57600, SERIAL_8E1);

  delay(500);

  Serial.println("Initialiseren motorsturing...");
  
  // De '3-berichten truc' om de poort/pijplijn stabiel te openen
  for(int i = 0; i < 3; i++) {
    motorOff();
    delay(200);
  }

  Serial.println("Motor inschakelen...");
  motorOn();
  delay(1000); // Geef de elektronica 1 seconde de tijd om te schakelen

/*while(1)
{
  setSpeed(100);
  delay(500);
  setSpeed(110);
  delay(500);
}*/
  
  pinMode(W5500_RST, OUTPUT);
  digitalWrite(W5500_RST, LOW);  delay(5);
  digitalWrite(W5500_RST, HIGH); delay(50);

  Ethernet.init(W5500_CS);
  Ethernet.setRstPin(W5500_RST);
  Ethernet.begin(mac, ip, dns, gw, sn);
  Udp.begin(localPort);


    
  delay(5000);

  Serial.println("\n[boot] up - Core 1 doet ALLES (Sensor+DAC)");
  Serial.print("IP: "); Serial.println(Ethernet.localIP());

  #ifdef TEST_MODE
    Serial.println("TEST MODE ON");
    memset(WIT, 0, 3600);
    memset(BLAUW, 0, 3600);
    WIT[30]=255;
    BLAUW[30]=255;
    //for(int w=0;w<1800;w++){WIT[w]=0;BLAUW[w+1800]=255;}
    //BLAUW[0] = 0xFF;
    //for(int w=0;w<3600;w++)WIT[w]=255;

  #endif
}

static void dumpRemainder(int rem) {
  uint8_t dump[256];
  while (rem > 0) {
    int n = Udp.read(dump, (rem < (int)sizeof(dump)) ? rem : (int)sizeof(dump));
    if (n <= 0) break;
    rem -= n;
  }
}

static void processFrameOnCore0_wit(void) {
  for (int i = 0; i < 1200; i++) { 
      WIT[i]      = Data[0][i];
      WIT[i+1200] = Data[1][i];
      WIT[i+2400] = Data[2][i];
  }
}

static void processFrameOnCore0_blauw(void) {
  for (int i = 0; i < 1200; i++) { 
      BLAUW[i]      = Data[3][i];
      BLAUW[i+1200] = Data[4][i];
      BLAUW[i+2400] = Data[5][i];
  }
}

void loop() {


  //TEST+=1;
  //delay(5);
  //TEST=1000;
  if(ISERNOGETHERNET>5){
    setSpeed(0);delay(1000);
    for (int i = 0; i < 3600; i++) { 
      WIT[i]   = 0;
      BLAUW[i] = 0;    
    }
    {
          ad7304_write_turbo(0, 0); 
          ad7304_write_turbo(1, 0);
          ad7304_write_turbo(2, 0); 
          ad7304_write_turbo(3, 0);
    }
  }
  

  // Simpele UDP loop, hoeft zich geen zorgen te maken over sensoren
  int pktSize = Udp.parsePacket();
  if (pktSize > 0) {
    uint8_t hdr[20];
    int needHdr = (pktSize < (int)sizeof(hdr)) ? pktSize : (int)sizeof(hdr);
    int h = Udp.read(hdr, needHdr);
    if (h < 0) h = 0;
    ISERNOGETHERNET=0;
/*
    /WS/blue0,b
    /WS/white0,b
    /WS/o/0
    0123456789
*/

    int8_t idx = -1;
    if (h >= 20 && hdr[1] == 'W' && hdr[4] == 'w' && (hdr[9] >= '0' && hdr[9] <= '2')) idx = (int8_t)(hdr[9] - '0');
    if (h >= 20 && hdr[1] == 'W' && hdr[4] == 'b' && (hdr[8] >= '0' && hdr[8] <= '2')) idx = (int8_t)(3+hdr[8] - '0');

    if (h < 20 && hdr[1] == 'W' && hdr[4] == 'o' && (hdr[6] == '0')) cor0=static_cast<int>(static_cast<signed char>(hdr[15])); //of zoiets h>=20 moet dus wel andersd
    if (h < 20 && hdr[1] == 'W' && hdr[4] == 'o' && (hdr[6] == '1')) cor2=static_cast<int>(static_cast<signed char>(hdr[15])); //of zoiets
    if (h < 20 && hdr[1] == 'W' && hdr[4] == 'o' && (hdr[6] == '2')) cor1=static_cast<int>(static_cast<signed char>(hdr[15])); //of zoiets
    if (h < 20 && hdr[1] == 'W' && hdr[4] == 'o' && (hdr[6] == '3')) cor3=static_cast<int>(static_cast<signed char>(hdr[15])); //of zoiets


    if (h < 20 && hdr[1] == 'W' && hdr[4] == 'r' && (hdr[6] == '0')) RPM=static_cast<unsigned int>(
    (static_cast<uint32_t>(hdr[12]) << 24) |  // MSB
    (static_cast<uint32_t>(hdr[13]) << 16) |
    (static_cast<uint32_t>(hdr[14]) << 8)  |
    (static_cast<uint32_t>(hdr[15]) << 0)    // LSB
); //of zoiets

    if(RPM<200)SLOW=true;
    else SLOW=false;
      if(SLOW==true)
 {
      //if (!haveArtnetMaster) return;

          
      //Serial.println("X");
  }
  //Serial1.println("X");*/

    if (idx < 0) { dumpRemainder(pktSize - h); drop_total++; return; }

    /*if((testteller%100)==0)Serial.println("TEST ");
  testteller++;*/
  //TEST++;if(TEST==3600)TEST=0;

    int need = 1200, got = 0;
    while (got < need) {
      int n = Udp.read(&Data[idx][got], need - got);
      if (n <= 0) break;
      got += n;
    }
    if (got < need) { dumpRemainder(pktSize - h - got); drop_total++; return; }

    int extra = pktSize - h - need;
    if (extra > 0) dumpRemainder(extra);

    recvMask |= (1u << idx);
    rx_total++;
    
    if (idx == 5) {
      #ifndef TEST_MODE
        processFrameOnCore0_wit();
        processFrameOnCore0_blauw();

        
          
        //Serial.println("process");

        if(NULPUNT)
        {
          NULPUNT=false;
          char msg[40];
          int len = snprintf(msg, sizeof(msg), "/WS/sensor/fall\n");
          if (len <= 0) return;
          if (len >= (int)sizeof(msg)) len = sizeof(msg) - 1;

           artnetMasterIP = Udp.remoteIP();
           //artnetMasterPort = Udp.remotePort();
          // haveArtnetMaster = true;

          Udp.beginPacket(artnetMasterIP, artnetMasterPort);
          Udp.write((const uint8_t*)msg, len);
          int result = Udp.endPacket();
          Serial.println(artnetMasterIP);

          

          if (result == 1) {
            
            //unsigned long tijdverlopen=vmillis-millis();
            //Serial.print(tijdverlopen);
            unsigned long nu=millis();
            unsigned long tijdverlopen=nu-vmillis;
            vmillis=nu;
            Serial.println(tijdverlopen);
            
          } else {
              Serial.println("Fout: Pakket kon niet worden verzonden.");
          }
          
        // Serial.println("X");
        }
        if(RPM!=vRPM)
        {
          Serial.print("niuwe RPM= ");
          Serial.println(RPM);
          setSpeed(RPM);
          vRPM=RPM;
        }
      #endif
      recvMask = 0; 
    }
  }
}
