FROM node:24-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:1.27-alpine

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/write-config.sh /docker-entrypoint.d/40-write-capswriter-config.sh
RUN chmod +x /docker-entrypoint.d/40-write-capswriter-config.sh
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
