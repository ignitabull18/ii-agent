if [ -n "$PROJECT_ID" ] && [ -n "$REGION" ]; then
  echo "Using Vertex Client"
  echo "PROJECT_ID: $PROJECT_ID"
  echo "REGION: $REGION"
  exec xvfb-run --auto-servernum python ws_server.py --project-id $PROJECT_ID --region $REGION
else
  echo "Using Anthropic Client, reading ANTHROPIC_API_KEY from .env"
  exec xvfb-run --auto-servernum python ws_server.py
fi
