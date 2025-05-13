#!/bin/bash

# .env 파일을 생성
echo "HF_TOKEN=$HF_TOKEN" > .env

# Vercel 로그인 및 배포 (환경변수 필요 시 --env 사용)
npx vercel --token=$VERCEL_TOKEN --prod