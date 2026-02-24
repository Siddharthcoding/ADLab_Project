# ----------------------------
# Build stage
# ----------------------------
FROM node:18-alpine AS build

WORKDIR /app

# Accept build-time variable from Render
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL

# Copy package files
COPY package*.json ./

# Install ALL dependencies (including dev)
RUN npm ci

# Copy source
COPY . .

# Build for production (VITE_API_URL will be injected here)
RUN npm run build


# ----------------------------
# Production stage
# ----------------------------
FROM nginx:alpine

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]