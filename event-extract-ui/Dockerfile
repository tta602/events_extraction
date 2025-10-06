FROM node:22-bullseye

WORKDIR /app

COPY package*.json ./

# Xóa node_modules + cache trước khi install
RUN npm ci --force --legacy-peer-deps

COPY . .

EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
