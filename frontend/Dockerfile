# Build phase
FROM node:20.12-alpine AS builder
WORKDIR /app

COPY package*.json ./
RUN npm ci   # <— no --omit=dev so Vite is available during build

COPY . .
RUN npm run build

# Serve phase
FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
