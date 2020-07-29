import pygame
import numpy as np
import sys
from threading import Thread
from test_digits import digit_rec, x_test
from math import pi, e, ceil
from time import sleep

pygame.init()

SC_SIZE = (1000, 700)
CV_SIZE = (560, 560)
CV_POS = (50, 50)
CV_FRAME = 10
DW_FRAME = 5
BT_FRAME = 3
DW_SIZE = (90, 90)
DW_POS = (760, 90)
PX_SIZE = (20, 20)
BT_SIZE = (70, 30)
CLEAR_BT_POS = (690, 580)
EXIT_BT_POS = (840, 580)
LINE_WIDTH = 20
FRAME_CLR = (198, 186, 255)
BUTTON_CLR = (215, 206, 255)
BUTTON_CLR_TRGT = (207, 198, 255)
BUTTON_CLR_PRS = FRAME_CLR
SCREEN_CLR = (237, 233, 255)
CANVAS_CLR = (255, 255, 255)
BLACK = (0, 0, 0)

exit_flag = False

font = pygame.font.Font(None, 30)
digit_font = pygame.font.Font(None, 100)
screen = pygame.display.set_mode(SC_SIZE)
cv_frame = pygame.Surface(tuple(map(lambda a, b: a+b, CV_SIZE, [CV_FRAME]*2)))
canvas = pygame.Surface(CV_SIZE)
dw_frame = pygame.Surface(tuple(map(lambda a, b: a+b, DW_SIZE, [DW_FRAME*2]*2)))
digit_window = pygame.Surface(DW_SIZE)
clear_button_frame = pygame.Surface(tuple(map(lambda a, b: a+b, BT_SIZE, [BT_FRAME*2]*2)))
clear_button = pygame.Surface(BT_SIZE)
clear_text = font.render('Clear', 1, BLACK)
exit_button_frame = pygame.Surface(tuple(map(lambda a, b: a+b, BT_SIZE, [BT_FRAME*2]*2)))
exit_button = pygame.Surface(BT_SIZE)
exit_text = font.render('Exit', 1, BLACK)

def draw(clear_button_color, exit_button_color, digit_rc):
    screen.fill(SCREEN_CLR)
    cv_frame.fill(FRAME_CLR)
    screen.blit(cv_frame, tuple(map(lambda a, b: a-b, CV_POS, [CV_FRAME]*2)))
    screen.blit(canvas, CV_POS)
    dw_frame.fill(FRAME_CLR)
    screen.blit(dw_frame, tuple(map(lambda a, b: a-b, DW_POS, [DW_FRAME]*2)))
    digit_window.fill(CANVAS_CLR)
    screen.blit(digit_window, DW_POS)
    clear_button_frame.fill(FRAME_CLR)
    screen.blit(clear_button_frame, tuple(map(lambda a, b: a-b, CLEAR_BT_POS, [BT_FRAME]*2)))
    clear_button.fill(clear_button_color)
    screen.blit(clear_button, CLEAR_BT_POS)
    screen.blit(clear_text, tuple(map(lambda a, b: a+b, CLEAR_BT_POS, [6, 5])))
    exit_button_frame.fill(FRAME_CLR)
    exit_button.fill(exit_button_color)
    screen.blit(exit_button_frame, tuple(map(lambda a, b: a-b, EXIT_BT_POS, [BT_FRAME]*2)))
    screen.blit(exit_button, EXIT_BT_POS)
    screen.blit(exit_text, tuple(map(lambda a, b: a+b, EXIT_BT_POS, [13, 5])))
    digit = digit_font.render(str(digit_rc), 1, BLACK)
    screen.blit(digit, tuple(map(lambda a, b: a+b, DW_POS, [25, 10])))

draw(BUTTON_CLR, BUTTON_CLR, '1')
canvas.fill(CANVAS_CLR)
pygame.display.update()

def color_function(dist):
    loc = 0
    scale = 15
    return tuple([(-e**-((dist-loc)**2/(2*scale**2))+1)*255]*3)

def draw_line(pos):
    x = pos[0]
    y = pos[1]
    x_min = max(50, x - LINE_WIDTH)
    x_max = min(610, x + LINE_WIDTH)
    y_min = max(50, y - LINE_WIDTH)
    y_max = min(610, y + LINE_WIDTH)
    x_pix = list(map(lambda a: a-CV_POS[0],
            list(range(ceil((x_min-PX_SIZE[0]//2)/PX_SIZE[0])*PX_SIZE[0]+PX_SIZE[0]//2, x_max, PX_SIZE[0]))))
    y_pix = list(map(lambda a: a-CV_POS[1],
            list(range(ceil((y_min-PX_SIZE[1]//2)/PX_SIZE[1])*PX_SIZE[1]+PX_SIZE[1]//2, y_max, PX_SIZE[1]))))
    for col in x_pix:
        for row in y_pix:
            dist = ((x-CV_POS[0]-col)**2+(y-CV_POS[1]-row)**2)**0.5
            cl = color_function(dist)
            if dist < LINE_WIDTH and cl[0] < cv_data[row//PX_SIZE[0], col//PX_SIZE[1]]:
                cv_data[row//PX_SIZE[0], col//PX_SIZE[1]] = cl[0]
                pygame.draw.rect(canvas, cl, (col, row, PX_SIZE[0], PX_SIZE[0]))


class Recognizer(Thread):
    def __init__(self, digit_hw):
        Thread.__init__(self)
        self.digit_hw = digit_hw
        self.digit_rc = None

    def run(self):
        while True:
            if exit_flag:
                sys.exit()
            self.digit_rc = digit_rec((255-self.digit_hw[np.newaxis, np.newaxis, :, :])/255)
            print(self.digit_rc)
            sleep(0.5)


cv_data = np.ones((CV_SIZE[0]//20, CV_SIZE[1]//20), dtype='float64') * 255
recognizer = Recognizer(cv_data)
recognizer.start()

while True:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            exit_flag = True
            sys.exit()

    pressed = pygame.mouse.get_pressed()
    pos = pygame.mouse.get_pos()
    if CV_POS[0] < pos[0] < CV_POS[0] + CV_SIZE[0] and CV_POS[1] < pos[1] < CV_POS[1] + CV_SIZE[1]:
        pygame.mouse.set_cursor(*pygame.cursors.broken_x)
        draw(BUTTON_CLR, BUTTON_CLR, recognizer.digit_rc)
    elif CLEAR_BT_POS[0] < pos[0] < CLEAR_BT_POS[0] + BT_SIZE[0] and CLEAR_BT_POS[1] < pos[1] < CLEAR_BT_POS[1] + BT_SIZE[1]:
        pygame.mouse.set_cursor(*pygame.cursors.tri_left)
        draw(BUTTON_CLR_TRGT, BUTTON_CLR, recognizer.digit_rc)
    elif EXIT_BT_POS[0] < pos[0] < EXIT_BT_POS[0] + BT_SIZE[0] and EXIT_BT_POS[1] < pos[1] < EXIT_BT_POS[1] + BT_SIZE[1]:
        pygame.mouse.set_cursor(*pygame.cursors.tri_left)
        draw(BUTTON_CLR, BUTTON_CLR_TRGT, recognizer.digit_rc)
    else:
        pygame.mouse.set_cursor(*pygame.cursors.arrow)
        draw(BUTTON_CLR, BUTTON_CLR, recognizer.digit_rc)
    if pressed[0]:
        if CV_POS[0] < pos[0] < CV_POS[0] + CV_SIZE[0] and CV_POS[1] < pos[1] < CV_POS[1] + CV_SIZE[1]:
            screen.fill(SCREEN_CLR)
            cv_frame.fill(FRAME_CLR)
            screen.blit(cv_frame, tuple(map(lambda a, b: a-b, CV_POS, [CV_FRAME]*2)))
            draw_line(pos)
            recognizer.digit_hw = cv_data
            screen.blit(canvas, CV_POS)
            dw_frame.fill(FRAME_CLR)
            screen.blit(dw_frame, tuple(map(lambda a, b: a-b, DW_POS, [DW_FRAME]*2)))
            digit_window = pygame.Surface(DW_SIZE)
            digit_window.fill(CANVAS_CLR)
            screen.blit(digit_window, DW_POS)
            clear_button_frame = pygame.Surface(tuple(map(lambda a, b: a+b, BT_SIZE, [BT_FRAME*2]*2)))
            clear_button_frame.fill(FRAME_CLR)
            screen.blit(clear_button_frame, tuple(map(lambda a, b: a-b, CLEAR_BT_POS, [BT_FRAME]*2)))
            clear_button = pygame.Surface(BT_SIZE)
            clear_button.fill(BUTTON_CLR)
            screen.blit(clear_button, CLEAR_BT_POS)
            clear_text = font.render('Clear', 1, BLACK)
            screen.blit(clear_text, tuple(map(lambda a, b: a+b, CLEAR_BT_POS, [5, 5])))
            exit_button_frame.fill(FRAME_CLR)
            exit_button.fill(BUTTON_CLR)
            screen.blit(exit_button_frame, tuple(map(lambda a, b: a-b, EXIT_BT_POS, [BT_FRAME]*2)))
            screen.blit(exit_button, EXIT_BT_POS)
            screen.blit(exit_text, tuple(map(lambda a, b: a+b, EXIT_BT_POS, [13, 5])))
            digit = digit_font.render(str(recognizer.digit_rc), 1, BLACK)
            screen.blit(digit, tuple(map(lambda a, b: a+b, DW_POS, [25, 10])))
        elif CLEAR_BT_POS[0] < pos[0] < CLEAR_BT_POS[0] + BT_SIZE[0] and CLEAR_BT_POS[1] < pos[1] < CLEAR_BT_POS[1] + BT_SIZE[1]:
            canvas.fill(CANVAS_CLR)
            draw(BUTTON_CLR_PRS, BUTTON_CLR, recognizer.digit_rc)
            cv_data *= 0
            cv_data += 255
            recognizer.digit_hw = cv_data
        elif EXIT_BT_POS[0] < pos[0] < EXIT_BT_POS[0] + BT_SIZE[0] and EXIT_BT_POS[1] < pos[1] < EXIT_BT_POS[1] + BT_SIZE[1]:
            draw(BUTTON_CLR, BUTTON_CLR_PRS, recognizer.digit_rc)
            exit_flag = True
            sys.exit()
    pygame.display.update()
