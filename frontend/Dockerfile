# ── Build stage ──────────────────────────────────────────
FROM node:20-alpine AS build
WORKDIR /app

# Install deps (cache layer)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# ── Serve stage ──────────────────────────────────────────
FROM nginx:alpine

# Install curl for healthcheck
RUN apk add --no-cache curl

# Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
