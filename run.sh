#!/bin/bash
IN=$1
arrIN=(${IN//./ })
nohup python -u "$1" > "${arrIN[0]}".log &