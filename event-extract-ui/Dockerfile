FROM node:22-bullseye

WORKDIR /app

# Chỉ copy package.json và package-lock.json
COPY package*.json ./

# Xóa cache npm trước
RUN npm cache clean --force

# Cài dependencies
RUN npm ci --force --legacy-peer-deps

# Copy code còn lại
COPY . .

EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--host"]
