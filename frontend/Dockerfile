FROM node:20-alpine AS build
WORKDIR /app

# Copy dependency manifests first
COPY frontend/package*.json ./
RUN npm ci

# Copy the rest of the frontend source
COPY frontend/. .

# Build production assets
RUN npm run build

# Production stage
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
