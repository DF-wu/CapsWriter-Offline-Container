import http from "node:http";

const port = Number(process.env.CAPSWRITER_MOCK_API_PORT || 6017);
const text = "mock transcript from CapsWriter Web Console";

function sendJson(res, status, body) {
  res.writeHead(status, {
    "Content-Type": "application/json;charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  });
  res.end(JSON.stringify(body));
}

function sendText(res, status, body, contentType = "text/plain;charset=utf-8") {
  res.writeHead(status, {
    "Content-Type": contentType,
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function formValue(body, name, fallback) {
  const match = body
    .toString("utf8")
    .match(new RegExp(`name="${name}"\\r\\n\\r\\n([^\\r]+)`));
  return match?.[1]?.trim() || fallback;
}

function transcriptionResponse(format) {
  if (format === "text") return { body: text, contentType: "text/plain;charset=utf-8" };
  if (format === "srt") {
    return {
      body: `1\n00:00:00,000 --> 00:00:01,500\n${text}\n`,
      contentType: "application/x-subrip;charset=utf-8",
    };
  }
  if (format === "vtt") {
    return {
      body: `WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n${text}\n`,
      contentType: "text/vtt;charset=utf-8",
    };
  }
  if (format === "verbose_json") {
    return {
      body: {
        task: "transcribe",
        language: "en",
        duration: 1.5,
        text,
        segments: [{ id: 0, start: 0, end: 1.5, text }],
        words: text.split(" ").map((word, index) => ({
          word,
          start: index * 0.2,
          end: index * 0.2 + 0.18,
        })),
      },
      contentType: "application/json;charset=utf-8",
    };
  }
  return { body: { text }, contentType: "application/json;charset=utf-8" };
}

const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    sendText(res, 204, "");
    return;
  }
  if (req.method === "GET" && req.url === "/health") {
    sendJson(res, 200, { status: "ok", model: "mock_asr", version: "dev" });
    return;
  }
  if (req.method === "GET" && req.url === "/ready") {
    sendJson(res, 200, {
      status: "ok",
      model: "mock_asr",
      version: "dev",
      checks: {
        task_router_bound: true,
        recognizer_process_alive: true,
        ffmpeg_available: true,
      },
      config: {
        auth_enabled: false,
        max_upload_mb: 100,
        task_timeout: 600,
        max_concurrent_requests: 2,
        cors_enabled: true,
        cors_origins_count: 1,
      },
    });
    return;
  }
  if (req.method === "GET" && req.url === "/v1/models") {
    sendJson(res, 200, {
      object: "list",
      data: [{ id: "mock_asr", object: "model", owned_by: "capswriter-offline", created: 0 }],
    });
    return;
  }
  if (req.method === "POST" && req.url === "/v1/audio/transcriptions") {
    const body = await readBody(req);
    const format = formValue(body, "response_format", "json");
    const response = transcriptionResponse(format);
    if (typeof response.body === "string") {
      sendText(res, 200, response.body, response.contentType);
    } else {
      sendJson(res, 200, response.body);
    }
    return;
  }
  sendJson(res, 404, { error: "not found" });
});

server.listen(port, "127.0.0.1", () => {
  console.log(`CapsWriter mock API listening on http://127.0.0.1:${port}`);
});
