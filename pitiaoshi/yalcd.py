import Adafruit_PCA9685
import time
import RPi.GPIO as GPIO
import spidev
import yah
class YALCD(object):
    def __init__(self):
        self.cs = 21
        self.rs = 20
        self.reset = 16
        self.temp = yah.YAH()
        self.asc16 = self.temp.ziku
        self.RED = self.temp.RED
        self.GREEN = self.temp.GREEN
        self.BLUE = self.temp.BLUE
        self.WHITE = self.temp.WHITE
        self.BLACK = self.temp.BLACK
        self.YELLOW = self.temp.YELLOW
        self.GRAY0 = self.temp.GRAY0
        self.GRAY1 = self.temp.GRAY1
        self.GRAY2 = self.temp.GRAY2
        self.spi = spidev.SpiDev()
        self.spi.open(0, 0)
        self.spi.max_speed_hz = 10000000

    def setByteData(self,input):
        lsb = input & 0xFF
        self.spi.writebytes([lsb])

    def Lcd_WriteIndex(self,cmd):
        GPIO.output(self.cs, False)
        GPIO.output(self.rs, False)
        self.setByteData(cmd)
        GPIO.output(self.cs, True)

    def Lcd_WriteData(self,data):
        GPIO.output(self.cs, False)
        GPIO.output(self.rs, True)
        self.setByteData(data)
        GPIO.output(self.cs, True)

    def yaini(self):
        GPIO.output(self.reset, 1)
        time.sleep(0.10)
        GPIO.output(self.reset, 0)
        time.sleep(0.10)
        GPIO.output(self.reset, 1)
        time.sleep(0.10)

        self.Lcd_WriteIndex(0x11)
        time.sleep(0.12)
        self.Lcd_WriteIndex(0xB1)
        self.Lcd_WriteData(0x01)
        self.Lcd_WriteData(0x2C)
        self.Lcd_WriteData(0x2D)
        self.Lcd_WriteIndex(0xB2)
        self.Lcd_WriteData(0x01)
        self.Lcd_WriteData(0x2C)
        self.Lcd_WriteData(0x2D)
        self.Lcd_WriteIndex(0xB3)
        self.Lcd_WriteData(0x01)
        self.Lcd_WriteData(0x2C)
        self.Lcd_WriteData(0x2D)
        self.Lcd_WriteData(0x01)
        self.Lcd_WriteData(0x2C)
        self.Lcd_WriteData(0x2D)
        self.Lcd_WriteIndex(0xB4)
        self.Lcd_WriteData(0x07)
        self.Lcd_WriteIndex(0xC0)
        self.Lcd_WriteData(0xA2)
        self.Lcd_WriteData(0x02)
        self.Lcd_WriteData(0x84)
        self.Lcd_WriteIndex(0xC1)
        self.Lcd_WriteData(0xC5)
        self.Lcd_WriteIndex(0xC2)
        self.Lcd_WriteData(0x0A)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteIndex(0xC3)
        self.Lcd_WriteData(0x8A)
        self.Lcd_WriteData(0x2A)
        self.Lcd_WriteIndex(0xC4)
        self.Lcd_WriteData(0x8A)
        self.Lcd_WriteData(0xEE)
        self.Lcd_WriteIndex(0xC5)
        self.Lcd_WriteData(0x0E)
        self.Lcd_WriteIndex(0x36)
        self.Lcd_WriteData(0xC8)
        self.Lcd_WriteIndex(0xe0)
        self.Lcd_WriteData(0x0f)
        self.Lcd_WriteData(0x1a)
        self.Lcd_WriteData(0x0f)
        self.Lcd_WriteData(0x18)
        self.Lcd_WriteData(0x2f)
        self.Lcd_WriteData(0x28)
        self.Lcd_WriteData(0x20)
        self.Lcd_WriteData(0x22)
        self.Lcd_WriteData(0x1f)
        self.Lcd_WriteData(0x1b)
        self.Lcd_WriteData(0x23)
        self.Lcd_WriteData(0x37)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(0x07)
        self.Lcd_WriteData(0x02)
        self.Lcd_WriteData(0x10)

        self.Lcd_WriteIndex(0xe1)
        self.Lcd_WriteData(0x0f)
        self.Lcd_WriteData(0x1b)
        self.Lcd_WriteData(0x0f)
        self.Lcd_WriteData(0x17)
        self.Lcd_WriteData(0x33)
        self.Lcd_WriteData(0x2c)
        self.Lcd_WriteData(0x29)
        self.Lcd_WriteData(0x2e)
        self.Lcd_WriteData(0x30)
        self.Lcd_WriteData(0x30)
        self.Lcd_WriteData(0x39)
        self.Lcd_WriteData(0x3f)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(0x07)
        self.Lcd_WriteData(0x03)
        self.Lcd_WriteData(0x10)

        self.Lcd_WriteIndex(0x2a)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(0x7f)

        self.Lcd_WriteIndex(0x2b)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(0x9f)
        self.Lcd_WriteIndex(0xF0)
        self.Lcd_WriteData(0x01)
        self.Lcd_WriteIndex(0xF6)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteIndex(0x3A)
        self.Lcd_WriteData(0x05)
        self.Lcd_WriteIndex(0x29)

    def Lcd_SetRegion(self,x_start, y_start, x_end, y_end):
        self.Lcd_WriteIndex(0x2A)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(x_start + 2)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(x_end + 2)

        self.Lcd_WriteIndex(0x2B)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(y_start + 3)
        self.Lcd_WriteData(0x00)
        self.Lcd_WriteData(y_end + 3)

        self.Lcd_WriteIndex(0x2C)

    def LCD_WriteData_16Bit(self,input):
        GPIO.output(self.cs, False)
        GPIO.output(self.rs, True)
        self.setByteData((input >> 8) & 0xFF)
        self.setByteData(input & 0xFF)
        GPIO.output(self.cs, True)

    def Lcd_Clear(self,Color):
        X_MAX_PIXEL = 128
        Y_MAX_PIXEL = 128
        self.Lcd_SetRegion(0, 0, X_MAX_PIXEL - 1, Y_MAX_PIXEL - 1)
        self.Lcd_WriteIndex(0x2C)
        for i in range(0, X_MAX_PIXEL):
            for m in range(0, Y_MAX_PIXEL):
                self.LCD_WriteData_16Bit(Color)

    def Gui_DrawPoint(self,x, y, fc):
        self.Lcd_SetRegion(x, y, x + 1, y + 1)
        self.LCD_WriteData_16Bit(fc)

    def Gui_DrawFont_GBK16(self,x, y, fc, bc, es):
        x0 = x
        for s in es:
            if (ord(s) < 128):
                k = ord(s)
                if (k == 13):
                    x = x0
                    y += 16
                else:
                    if (k > 32):
                        k -= 32
                    else:
                        k = 0
                    for i in range(0, 16):
                        for j in range(0, 8):
                            if (self.asc16[k * 16 + i] & (0x80 >> j)):
                                self.Gui_DrawPoint(x + j, y + i, fc)
                            else:
                                if (fc != bc):
                                    self.Gui_DrawPoint(x + j, y + i, bc)
                    x = x + 8
                j = j + 1