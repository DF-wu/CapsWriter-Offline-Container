FROM node:24-alpine@sha256:a0b9bf06e4e6193cf7a0f58816cc935ff8c2a908f81e6f1a95432d679c54fbfd AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:1.27-alpine@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10

RUN mkdir -p /etc/nginx/snippets
COPY deploy/security-headers.conf /etc/nginx/snippets/capswriter-security-headers.conf
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/write-config.sh /docker-entrypoint.d/40-write-capswriter-config.sh
RUN chmod +x /docker-entrypoint.d/40-write-capswriter-config.sh
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
