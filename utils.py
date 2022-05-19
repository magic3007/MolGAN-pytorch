#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : utils.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 05.19.2022
# Last Modified Date: 05.19.2022
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>

import datetime
import logging
import time

log = logging.getLogger(__name__)


def get_date_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class DeltaTimeFormatter(logging.Formatter):
    def format(self, record):
        duration = datetime.datetime.utcfromtimestamp(record.relativeCreated / 1000)
        record.delta = duration.strftime("%H:%M:%S")
        return super().format(record)


def log_everything(log_file_name):
    # Reference
    # *** Print the elapsed time in log ***
    # https://stackoverflow.com/questions/25194864/python-logging-time-since-start-of-program
    # *** Print to screen and log file at the same time ***
    # https://stackoverflow.com/questions/9321741/printing-to-screen-and-writing-to-a-file-at-the-same-time
    # https://stackoverflow.com/questions/13733552/logger-configuration-to-log-to-file-and-print-to-stdout

    LOGFORMAT = '%(asctime)-12s +%(delta)s %(pathname)s:%(lineno)d [%(levelname)-8s] - %(message)s'
    log_formatter = DeltaTimeFormatter(LOGFORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # set up logging to file
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # define a Handler which writes to the sys.stderr
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


if __name__ == '__main__':
    print(get_date_str())
    log_everything('/tmp/test.log')
    logging.info("hello world!")
    time.sleep(1)
    logging.info("hello world!")
    time.sleep(1)
    log.info("hello world!")
