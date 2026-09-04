# Third-Party Notices and Resource Approval

Status: Phase 1 personal-use record; redistribution is not approved or reviewed.

This project uses free/open-source software and locally installed speech resources. "Free to download" and "open source" do not mean every binary, model, dataset, or voice has identical attribution, commercial-use, or redistribution rights.

Personal local development may begin before this file is complete. Complete the applicable redistribution sections before sharing a packaged build or bundled models with another person.

## 1. Application

```text
Application licence: unset (personal-use-only project; no application licence file)
Copyright holder:
Repository/source location:
```

## 2. Production Python dependencies

Record exact versions after compatibility testing and lockfile creation.

| Component | Version | Licence | Official source | Attribution/redistribution notes |
| --- | --- | --- | --- | --- |
| FastAPI | 0.139.2 | MIT | https://github.com/fastapi/fastapi | review before redistribution |
| Uvicorn | 0.51.0 | BSD-3-Clause | https://github.com/encode/uvicorn | review before redistribution |
| pypdf | TBD | TBD | TBD | review after pinning |
| pdfplumber | TBD | TBD | TBD | review after pinning |
| sentence segmenter | TBD | TBD | TBD | selected during implementation |
| selected TTS package | TBD | TBD | TBD | Phase 0 decides |
| other native/runtime dependencies | TBD | TBD | TBD | add all applicable items |

Add transitive/native items that carry attribution or redistribution obligations when preparing a distributable package.

## 3. Selected TTS engine

```text
Name:
Version or commit:
Licence:
Official source URL:
Package or source checksum if applicable:
Personal-use restrictions:
Redistribution notes:
Required attribution:
```

Phase 0 selected Kokoro 0.9.4 for personal local use. Redistribution of the application and bundled resources is not approved or reviewed.

### Kokoro candidate

```text
Name: Kokoro
Version: kokoro 0.9.4
Licence: Apache-2.0 (package and Kokoro-82M model)
Official source URL: https://github.com/hexgrad/kokoro
Model ID: hexgrad/Kokoro-82M
Model revision/checksum: captured-at-download; unresolved until downloaded
Personal-use restrictions: none identified in this record; verify the exact resource
Redistribution notes: review package, model, voice data, and espeak-ng build together
Required attribution: Apache-2.0 notice and any included upstream notices
Native dependency note: Windows phonemization may require espeak-ng; exact build TBD
```

### Chatterbox Nano

```text
Name: Chatterbox Nano
Version: chatterbox-tts 0.1.7 when sufficient; otherwise official source commit
5de7a54aa4e5e2baadb0182dde554908b48b85c2 for Nano support
Licence: MIT (source and Nano model card)
Official source URL: https://github.com/resemble-ai/chatterbox
Model ID: ResembleAI/chatterbox-nano
Model revision/checksum: source commit above; model files are cached locally by the Chatterbox environment
Personal-use restrictions: review the exact model card and bundled resources
Redistribution notes: users must own or have permission to use any uploaded reference voice; the app stores it only in the workspace
Required attribution: MIT notice and any included upstream notices
```

## 4. Selected model

```text
Model:
Version or revision:
Licence:
Official source URL:
SHA-256:
Source dataset restrictions reviewed: yes/no
Personal-use restrictions:
Redistribution notes:
Required attribution:
```

The model fields above are intentionally unresolved until an official run captures a
revision and SHA-256. Do not represent a model download as checksum-verified before
that capture.

## 5. Approved voices

Four voices are approved for personal local use. Static candidate listings are not approvals. Kokoro's
English built-ins include `af_heart`, `af_bella`, `af_nicole`, `af_aoede`, `af_kore`,
`af_sarah`, `af_nova`, `af_sky`, `af_alloy`, `af_jessica`, `af_river`, `am_michael`,
`am_fenrir`, `am_puck`, and `am_echo`; Chatterbox exposes its built-in/default voice
and an explicit reference-WAV option. The approved Kokoro voices are
`af_heart`, `af_bella`, `bf_emma`, and `bf_isabella`. A reference WAV or dataset must not be bundled until its source,
licence, attribution, and redistribution permission are resolved.

### Voice 1

```text
Internal ID:
Display name:
Source:
Version or revision:
Voice/model licence:
Dataset licence if separate:
Attribution required:
Redistribution allowed:
Commercial-use notes if relevant:
SHA-256:
```

### Voice 2

```text
Internal ID:
Display name:
Source:
Version or revision:
Voice/model licence:
Dataset licence if separate:
Attribution required:
Redistribution allowed:
Commercial-use notes if relevant:
SHA-256:
```

### Voice 3

```text
Internal ID:
Display name:
Source:
Version or revision:
Voice/model licence:
Dataset licence if separate:
Attribution required:
Redistribution allowed:
Commercial-use notes if relevant:
SHA-256:
```

## 6. FFmpeg and ffprobe

Do not record only "FFmpeg is free." The enabled components and build configuration affect the applicable terms.

```text
Distribution/build name:
Version:
Official source URL:
Build configuration or source information:
Licence applicable to this exact build:
SHA-256 ffmpeg.exe:
SHA-256 ffprobe.exe:
Required notices or source-offer obligations:
```

## 7. Resource manifest

`resources/manifest.json` should contain revision-pinned source locations and checksums for non-Python resources.

Example:

```json
{
  "schema_version": 1,
  "resources": [
    {
      "id": "voice-1",
      "url": "revision-pinned-source",
      "sha256": "...",
      "licence": "...",
      "attribution": "..."
    }
  ]
}
```

Do not automatically substitute another model or binary when a checksum fails.

## 8. Personal use versus distribution

Immediate goal: personal local use.

Personal testing of locally downloaded resources is distinct from redistribution
approval. It does not grant permission to bundle model weights, voice files,
reference recordings, datasets, or native binaries in a packaged application.

Before sharing a prepackaged executable, bundled model, or FFmpeg build:

- pin exact dependency versions
- record the selected TTS engine, model, and all four voices
- record the exact FFmpeg build
- include required attribution
- record checksums
- resolve unclear redistribution terms or replace the resource

## 9. Distribution gate

A packaged build must not be shared until all applicable fields above are complete and the exact included resources are approved.
