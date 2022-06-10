import Adafruit_PCA9685
import RPi.GPIO as GPIO
import yalcd
#led = 4
def set_servo_angle(channel, angle):
    date = int(4096 * ((angle * 11) + 500) / (20000) + 0.5)
    pwm.set_pwm(channel, 0, date)

def yago():
    GPIO.output(26, 0)
    GPIO.output(19, 1)
    GPIO.output(13, 0)
    GPIO.output(6, 1)

def yastop():
    GPIO.output(26, 0)
    GPIO.output(19, 0)
    GPIO.output(13, 0)
    GPIO.output(6, 0)

if __name__ == '__main__':
    lcd=yalcd.YALCD()
    GPIO.setmode(GPIO.BCM)
    # GPIO.setup(led, GPIO.IN)
    GPIO.setup(21, GPIO.OUT)
    GPIO.setup(20, GPIO.OUT)
    GPIO.setup(16, GPIO.OUT)

    GPIO.setup(26, GPIO.OUT)  # 1
    GPIO.setup(19, GPIO.OUT)
    GPIO.setup(13, GPIO.OUT)
    GPIO.setup(6, GPIO.OUT)  # 4

    GPIO.setup(5, GPIO.OUT)
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(50)
    set_servo_angle(0, 90)
    pwm.set_pwm_freq(50)
    set_servo_angle(1, 90)
    GPIO.output(5, 1)
    yastop()

    lcd.yaini()
    lcd.Lcd_Clear(lcd.BLACK)

    print("start")
    while 1:
        a2 = input("") # 0 1
        a1 = int(a2)
        if a1 <= 180 and a1 > 0:
            # che = int(input("che:"))#0 1
            channel = 0  # 通道
            angle = a1  # 角度
            pwm.set_pwm_freq(50)  # 频率
            set_servo_angle(channel, angle)

            channel = 1  # 通道
            angle = a1  # 角度
            pwm.set_pwm_freq(50)  # 频率
            set_servo_angle(channel, angle)
            yago()
            GPIO.output(5, 0)
            l1 = 'jiaodu'
            l2 = list(l1)
            lcd.Gui_DrawFont_GBK16(10, 50, lcd.BLACK, lcd.YELLOW, l2)

            l1 = a2
            l2 = list(l1)
            if(a1<10):
                ltemp=['0']
                ltemp.append(l2[0])
                lcd.Gui_DrawFont_GBK16(10, 70, lcd.BLACK, lcd.YELLOW, ltemp)
            else:
                lcd.Gui_DrawFont_GBK16(10, 70, lcd.BLACK, lcd.YELLOW, l2)
        else:
            print("errno")
            yastop()
            GPIO.output(5, 1)
            lcd.Lcd_Clear(lcd.BLACK)