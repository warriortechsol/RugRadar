# frontend/Dockerfile
FROM node:20-alpine AS build
WORKDIR /app

# Copy manifests first for better caching
COPY package.json package-lock.json* ./

# Deterministic if lockfile exists; otherwise proceed with npm install
RUN if [ -f package-lock.json ]; then npm ci; else echo 'No package-lock.json found; using npm install (non-deterministic)'; npm install; fi

COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
