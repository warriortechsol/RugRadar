FROM node:20-alpine AS builder
WORKDIR /app

# Copy only frontend package manifests first (better layer caching)
COPY frontend/package*.json ./

# Install deps
RUN npm ci

# Copy rest of the frontend files
COPY frontend/. .

# Build
RUN npm run build

# Serve via nginx
FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
