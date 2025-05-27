#!/bin/sh

# Substitute environment variables in template
echo "Substituting environment variables in template"
echo "HOST_IP: ${HOST_IP}"
export ESCAPED_HOST_IP=$(echo "${HOST_IP}" | sed 's/\./\\./g')
echo "ESCAPED_HOST_IP: ${ESCAPED_HOST_IP}"
envsubst '${ESCAPED_HOST_IP}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

# Test nginx configuration
exec nginx -g 'daemon off;'