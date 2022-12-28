#pragma once
#include <iostream>
#include <ostream>
#include <string>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>
#include "macro.h"

class progressbar
{

    public:
        inline progressbar(int cycle_num, std::string message_ = "")
            : progress(0),
              n_cycles(cycle_num),
              last_perc(0),
              done_char("█"),
              todo_char(" "),
              opening_bracket_char("["),
              closing_bracket_char("]"),
              output(std::cerr)
        {
            // get terminal width
            struct winsize w;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
            terminal_width = w.ws_col;
            bar_width = terminal_width / 2;
            message = message_;
            start_time = CURRENT_TIME;
        }

        // main function
        inline void update()
        {
            int perc = progress * bar_width / (n_cycles - 1);
            ++progress;
            if (perc <= last_perc)
                return;
            // back to beginning of line
            output << std::string(terminal_width, '\b');
            // write opening bracket
            output << opening_bracket_char;
            // write done characters
            int n_done = perc;
            for (int i = 0; i < n_done; ++i)
                output << done_char;
            // write todo characters
            int n_todo = bar_width - n_done;
            for (int i = 0; i < n_todo; ++i)
                output << todo_char;
            // readd trailing percentage characters
            std::string perc_str = std::to_string(std::min(progress * 100 / n_cycles, 100));
            output << closing_bracket_char << ' ' << perc_str << '%' << std::string(3 - perc_str.size(), ' ');
            int seconds =
                std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start_time)
                    .count();
            // format time using hh:mm:ss
            int hours = seconds / 3600;
            seconds -= hours * 3600;
            int minutes = seconds / 60;
            seconds -= minutes * 60;
            output << ' ' << std::string(2 - std::to_string(hours).size(), '0') << hours << ':'
                   << std::string(2 - std::to_string(minutes).size(), '0') << minutes << ':'
                   << std::string(2 - std::to_string(seconds).size(), '0') << seconds;
            // write message
            output << ' ' << message;
            last_perc = perc;
            output << std::flush;
            return;
        }

        inline int get_progress() { return progress; }

    private:
        int progress;
        int n_cycles;
        int last_perc;
        int bar_width;
        int terminal_width;
        std::string done_char;
        std::string todo_char;
        std::string opening_bracket_char;
        std::string closing_bracket_char;
        std::string message;
        std::chrono::steady_clock::time_point start_time;

        std::ostream &output;
};
