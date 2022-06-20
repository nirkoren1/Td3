import time
import winsound


class RecordHelper:
    def __init__(self):
        self.start_time = time.time()
        self.render_windows = [(0 + i * 20, 10 + i * 20) for i in range(50)]
        self.beeped = [False for i in self.render_windows]

    def should_render(self):
        time_value = (time.time() - self.start_time) / 60
        for index, window in enumerate(self.render_windows):
            if window[0] <= time_value <= window[1] and not self.beeped[index]:
                if not self.beeped[index]:
                    self.beeped[index] = True
                    winsound.Beep(1000, 6000)
                    input("   --press any key to start rendering--")
                    winsound.Beep(1000, 2000)
                return True
        return False

    def print_time(self, loop):
        print('\r', str(loop) + " " + time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time)), end='')

    def render_loop(self, loop, env, ag, observation):
        render_start = time.time()
        if self.should_render():
            render_loop_start = time.time()
            while (time.time() - render_loop_start) / 60 <= 1.25:
                print('\r', str(loop) + " " + str(int((time.time() - render_loop_start) / 60)) + ":" + str(
                    int((time.time() - render_loop_start) % 60)), end='')
                env.render()
                action = ag.take_an_action_for_real(observation)
                observation, reward, done, info = env.step(action)
                if done:
                    observation = env.reset()
            self.start_time += time.time() - render_start
