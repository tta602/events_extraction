# frontend Dockerfile
FROM node:22

WORKDIR /app

COPY package*.json ./

# Xóa cache cũ để tránh lỗi optional dependencies
RUN npm ci --legacy-peer-deps

COPY . .

EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--host"]
