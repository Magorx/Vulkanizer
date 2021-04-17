class TrackerFPS {
	double last_time;
	double frame_time;
	size_t frame_cnt;

public:
	void init() {
		last_time = glfwGetTime();
		frame_time = 0;
		frame_cnt = 0;
	}

	bool tick() {
		double time = glfwGetTime();
		frame_time += time - last_time;
		last_time = time;
		++frame_cnt;

		if (frame_time > 1.0) {
			frame_time -= 1;
			printf("[FPS]: %lu\n", frame_cnt);
			frame_cnt = 0;
			return true;
		}

		return false;
	}
};
