// ==========================================
// CONFIGURATIE
// ==========================================
//#define TEST_MODE // Uncomment voor testmodus
#define DEBUG_SERIAL // Uncomment voor logging in het pakketpad (kost UDP pakketten!)
//nog doen. Wanneer RPM==0 de 4 DACS data halen uit White[0],White[1800] en blu... etc

// ETHERNET_LARGE_BUFFERS splits the W5500's 16 KB of RX memory over MAX_SOCK_NUM sockets instead
// of using the fixed 2 KB default. Ethernet_Generic is header-only, so this define does reach the
// library from here.
#define ETHERNET_LARGE_BUFFERS 1
// NOTE: setting MAX_SOCK_NUM here has NO effect on RP2040 — Ethernet_Generic.hpp #undef's it (this
// board has no RAMEND/RAMSTART, so it takes the #undef branch) and ETHERNET_LARGE_BUFFERS then
// pins it at 2 unconditionally. 2 sockets => SSIZE = 8192, so our one UDP socket gets 8 KB RX.
// The value below only documents what the library ends up using; do not "tune" it. One frame is
// 6 chunk datagrams of 1220 bytes = 7368 bytes with the chip's 8-byte per-datagram headers, so
// there is ~11% headroom — see the boot report in setup() to verify against the actual chip.
#define MAX_SOCK_NUM 2
volatile int ISERNOGETHERNET=10;   // volatile: incremented on core 1, cleared on core 0

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
#include <string.h>
#include <ws_protocol.h>
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
// Double buffered: core 0 fills the back buffer and then swaps the pointer, so core 1 never reads
// an array that is halfway through a 3600-byte memcpy.
static uint8_t WIT_BUF[2][3600];
static uint8_t BLAUW_BUF[2][3600];
// `volatile` applies to the pointer, not the pixels: core 1 must re-read it every sample so it
// picks up the swap, but the buffer it points at is never written while it is the published one.
static uint8_t * volatile WIT   = WIT_BUF[0];
static uint8_t * volatile BLAUW = BLAUW_BUF[0];
static uint8_t backBuffer = 1;
static uint8_t recvMask = 0;
// All six chunks of a frame present: /WS/white0..2 (bits 0-2) and /WS/blue0..2 (bits 3-5).
#define FRAME_COMPLETE 0x3F
// Upper bound on datagrams drained per loop() call — two frames' worth. Only ever reached under a
// packet flood (port 6454 is also the Art-Net port); it keeps the watchdog from being starved.
#define MAX_DRAIN_PER_LOOP 12
static uint32_t rx_total = 0, drop_total = 0, incomplete_total = 0;

IPAddress artnetMasterIP(192,168,1,123); // fallback, change if useful
uint16_t artnetMasterPort = 6454;
bool haveArtnetMaster = false;

// ==========================================
// CORE 1: ALLES-IN-EEN (DAC + SENSOR)
// ==========================================

// Variabelen voor PLL (Alles float voor snelheid)
#define MIN_INTERVAL 500
#define MAX_INTERVAL 1000000
// Weight of ONE new measurement in the interval estimate. MEASURED: 0.95 (the original) makes
// `pll_interval` the raw previous-revolution time, so every bit of edge-timing noise is written
// straight into this revolution's pixel spacing — visible as jitter on EVERY frame. 0.05 averages
// it over ~20 revolutions and the ring is stable. Speed changes are handled by the phase-error
// term below, not by this; that is the usual PLL split (slow frequency, fast phase). Do not raise.
#define ALPHA 0.05f // 'f' forceert float
// How far past the end of the ring the sample counter may run, and the phase error the step
// correction will act on in one revolution (samples; 3600 = one full turn).
#define WTELLER_CEILING 4320
#define MAX_PHASE_ERROR 360.0f

float pll_interval = 30000.0f;
float step = 8.33f; // Startwaarde
// Sample count reached at the last sync edge, for the DEBUG_SERIAL report. Written on core 1,
// read on core 0 — diagnostics only, so a torn read would just misprint one line.
volatile int wteller_at_sync = 0;
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

// De Modbus CRC-16 berekening leeft in lib/ws_protocol/ws_protocol.h zodat `pio test` hem kan
// testen zonder dit bestand te compileren.

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
          // Wteller runs past 3600 (see below), so this is SIGNED: positive = too slow to finish
          // the ring, negative = finished early and idled. That sign is what makes this a closed
          // loop. While the counter saturated at 3600 the error read 0 on every revolution once
          // the ring was being finished, so there was no feedback at all and `step` was just a
          // copy of the last measured period. Clamped so one wild revolution (missed edge, sensor
          // glitch) cannot drive the divisor near zero.
          float error = 3600.0f - (float)Wteller;
          if (error >  MAX_PHASE_ERROR) error =  MAX_PHASE_ERROR;
          if (error < -MAX_PHASE_ERROR) error = -MAX_PHASE_ERROR;

          // Pas de stapgrootte aan: als error > 0 (te laat), maak stapjes groter.
          // We corrigeren een klein deel van de fout (bijv. 10%) per ronde
          float correction = (error * 0.10f);
          step = (pll_interval / (3600.0f + correction));
        }
      }
      start_time = now;

      // Harde Sync blijft nodig om jitter te voorkomen
      next_t_us = (double)now;
      wteller_at_sync = Wteller;   // how far the ring got this revolution (3600 = saturated)
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
          // Snapshot both pointers once so all four lamps of this sample come from the same
          // frame even if core 0 swaps buffers mid-sample (and so it is one load, not four).
          uint8_t *wit   = WIT;
          uint8_t *blauw = BLAUW;
          ad7304_write_turbo(0, wit[(Wteller + 0+TEST+cor0) % 3600]);
          ad7304_write_turbo(1, blauw[(Wteller + 1800+900+TEST+cor1) % 3600]);
          ad7304_write_turbo(2,wit[(Wteller + 1800 + TEST+cor2) % 3600]);
          ad7304_write_turbo(3, blauw[(Wteller + 900+TEST+cor3) % 3600]);
    /*
          ad7304_write_turbo(0, 255);
          ad7304_write_turbo(1, 255);
          ad7304_write_turbo(2,255);
          ad7304_write_turbo(3, 255);
    */

        }
        // Keep counting past the end of the ring (up to a ceiling) so the phase error at the next
        // edge can go negative. Capping the counter at 3600 kills the feedback: the error then
        // reads 0 forever and `step` follows the raw last measurement, which jitters every frame.
        if (Wteller < WTELLER_CEILING) Wteller++;
      }

    }

    else
      {
          uint8_t *wit   = WIT;
          uint8_t *blauw = BLAUW;
          ad7304_write_turbo(0, wit[0]);
          ad7304_write_turbo(1, blauw[0]);
          ad7304_write_turbo(2,wit[1800]);
          ad7304_write_turbo(3, blauw[1800]);
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

  // Socket buffer report — read back from the W5500 itself, not from our #defines, because
  // Ethernet_Generic overrides MAX_SOCK_NUM on this board. Expect: sockets=2, SSIZE=8192,
  // SnRX_SIZE=8 KB, chip=4 (w5500). One frame needs 7368 bytes, so RX must be >= 8 KB.
  Serial.print("[w5500] chip="); Serial.print((int)W5100.getChip());
  Serial.print(" sockets="); Serial.print(MAX_SOCK_NUM);
  Serial.print(" SSIZE="); Serial.print(W5100.SSIZE);
  Serial.print(" SnRX_SIZE(0)="); Serial.print(W5100.readSnRX_SIZE(0));
  Serial.print("K SnTX_SIZE(0)="); Serial.print(W5100.readSnTX_SIZE(0));
  Serial.print("K  frame needs "); Serial.print(6 * (1220 + 8));
  Serial.println(" bytes");

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

// Fill the back buffer from the six staged chunks, then publish it to core 1 with a single
// pointer store (aligned, so atomic on RP2040). Core 1 keeps reading the old buffer until then.
static void processFrameOnCore0(void) {
  uint8_t *wit   = WIT_BUF[backBuffer];
  uint8_t *blauw = BLAUW_BUF[backBuffer];

  for (int i = 0; i < 1200; i++) {
      wit[i]      = Data[0][i];
      wit[i+1200] = Data[1][i];
      wit[i+2400] = Data[2][i];
  }
  for (int i = 0; i < 1200; i++) {
      blauw[i]      = Data[3][i];
      blauw[i+1200] = Data[4][i];
      blauw[i+2400] = Data[5][i];
  }

  __sync_synchronize();   // buffer writes must land before core 1 can see the new pointer
  WIT        = wit;
  BLAUW      = blauw;
  backBuffer ^= 1;
}

void loop() {


  //TEST+=1;
  //delay(5);
  //TEST=1000;
  #ifdef DEBUG_SERIAL
    // Once per second, outside the packet path. `incomplete` counts frames that arrived with a
    // missing chunk — if it climbs, the host is still outrunning the socket buffer.
    {
      static unsigned long lastReport = 0;
      unsigned long nowMs = millis();
      if (nowMs - lastReport >= 1000) {
        lastReport = nowMs;
        Serial.print("rx="); Serial.print(rx_total);
        Serial.print(" incomplete="); Serial.print(incomplete_total);
        Serial.print(" pll="); Serial.print(pll_interval);
        Serial.print(" Wteller="); Serial.println(wteller_at_sync);
      }
    }
  #endif

  // Ethernet watchdog: no packets for several revolutions → stop the motor and blank the lamps.
  // Blanking goes through the back buffer so core 1 keeps sole ownership of the SPI1 DAC bus, and
  // the retry is rate limited instead of blocking the UDP drain with delay(1000).
  if(ISERNOGETHERNET>5){
    static unsigned long lastWatchdog = 0;
    unsigned long nowMs = millis();
    if (nowMs - lastWatchdog >= 1000) {
      lastWatchdog = nowMs;
      setSpeed(0);
      memset(WIT_BUF[backBuffer],   0, 3600);
      memset(BLAUW_BUF[backBuffer], 0, 3600);
      __sync_synchronize();
      WIT        = WIT_BUF[backBuffer];
      BLAUW      = BLAUW_BUF[backBuffer];
      backBuffer ^= 1;
      SLOW = true;   // stop stepping the POV image; core 1 falls back to WIT[0]/WIT[1800]
    }
  }


  // Simpele UDP loop, hoeft zich geen zorgen te maken over sensoren.
  // Drain the whole socket queue here rather than one datagram per loop() call: the host sends a
  // frame as six 1220-byte chunks and every bail-out below used to cost a full loop() iteration,
  // so config packets alone could stall the drain long enough to overflow the 8 KB RX buffer.
  int pktSize;
  int drained = 0;
  while (drained++ < MAX_DRAIN_PER_LOOP && (pktSize = Udp.parsePacket()) > 0) {
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

    if (idx < 0) { dumpRemainder(pktSize - h); drop_total++; continue; }

    /*if((testteller%100)==0)Serial.println("TEST ");
  testteller++;*/
  //TEST++;if(TEST==3600)TEST=0;

    int need = 1200, got = 0;
    while (got < need) {
      int n = Udp.read(&Data[idx][got], need - got);
      if (n <= 0) break;
      got += n;
    }
    if (got < need) { dumpRemainder(pktSize - h - got); drop_total++; continue; }

    int extra = pktSize - h - need;
    if (extra > 0) dumpRemainder(extra);

    recvMask |= (1u << idx);
    rx_total++;

    if (idx == 5) {
      #ifndef TEST_MODE
        // Only publish a frame we received in full. An incomplete frame used to be committed
        // anyway, which put the previous frame's pixels in that third of the ring for one
        // revolution — the white light appearing to shift for a single frame. Holding the last
        // complete frame for one more revolution is far less visible.
        if (recvMask == FRAME_COMPLETE) {
          processFrameOnCore0();
        } else {
          incomplete_total++;
        }

        if(NULPUNT)
        {
          NULPUNT=false;
          static const char msg[] = "/WS/sensor/fall\n";
          const int len = (int)sizeof(msg) - 1;

           artnetMasterIP = Udp.remoteIP();
           //artnetMasterPort = Udp.remotePort();
          // haveArtnetMaster = true;

          Udp.beginPacket(artnetMasterIP, artnetMasterPort);
          Udp.write((const uint8_t*)msg, len);
          int result = Udp.endPacket();

          // Printing here blocks core 0 at 115200 baud exactly while the next burst is landing,
          // which costs us chunks — keep it out of the packet path unless explicitly debugging.
          unsigned long nu = millis();
          #ifdef DEBUG_SERIAL
            Serial.print(result == 1 ? "fall " : "fall SEND FAILED ");
            Serial.println(nu - vmillis);
          #else
            (void)result;
          #endif
          vmillis = nu;
        }
        if(RPM!=vRPM)
        {
          #ifdef DEBUG_SERIAL
            Serial.print("niuwe RPM= ");
            Serial.println(RPM);
          #endif
          setSpeed(RPM);
          vRPM=RPM;
        }
      #endif
      recvMask = 0;
    }
  }
}
