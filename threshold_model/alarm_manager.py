import pygame


class AlarmManager:
    def __init__(self, alarm_path):
        pygame.mixer.init()
        self.alarm_path = alarm_path
        self.is_playing = False

    def start_alarm(self):
        if not self.is_playing:
            pygame.mixer.music.load(self.alarm_path)
            pygame.mixer.music.play(-1)   # loop forever
            self.is_playing = True

    def stop_alarm(self):
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False