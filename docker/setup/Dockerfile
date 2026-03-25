FROM alpine:3.19

RUN apk add --no-cache curl jq

COPY healthcheck.sh /healthcheck.sh
RUN chmod +x /healthcheck.sh

CMD ["/healthcheck.sh"]
