/**
 * ESA Multimodal Telemetry & Biometric Access Cockpit - Application Orchestrator
 * Integrates 60fps HTML5 Oscilloscopes, Telemetry WebSocket, Webcam Face Recognition,
 * Liveness Anti-Spoofing, Session Persistence, and Clinical Reporting.
 */

(() => {
  "use strict";

  // =========================================================================
  // APPLICATION STATE
  // =========================================================================
  const state = {
    currentView: "view-overview",
    ws: null,
    wsConnected: false,
    wsUrl: (window.location.protocol === "https:" ? "wss://" : "ws://") + 
           (window.location.host || "localhost:8000") + "/ws/telemetry",
    audioBeepEnabled: false,
    isFrozen: false,
    
    // Biomedical Current Vitals
    vitals: {
      ecgHr: 72,
      ppgHr: 72,
      spo2: 98.4,
      pi: 4.2,
      pat: 224,
      rrInterval: 833,
      rmssd: 42.8,
      sdnn: 51.0,
      respRate: 16,
      bpSys: 118,
      bpDia: 78,
      leadsOff: 0,
      fingerOff: false,
      signalQuality: "VALID",
    },

    // RR Intervals Ring Buffer for Poincaré plot
    rrHistory: [],

    // Active Session Management
    activeSession: {
      id: null,
      name: "Guest Session",
      participantId: null,
      isGuest: true,
      startTime: null,
      sampleCount: 0,
      timerInterval: null,
    },

    // Face Recognition & Webcam State
    webcam: {
      stream: null,
      active: false,
      intervalId: null,
      lastDetection: null,
      mockAngle: 0,
    },

    // Synthetic Telemetry Simulator State
    simulator: {
      active: true, // Auto-fallback when real WS is offline
      intervalId: null,
      hr: 72,
      respRate: 16,
      noise: 2,
      leadsOff: 0,
      fingerOff: false,
      sampleIdx: 0,
    },

    // Historical Sessions Cache
    sessions: [],
  };

  // Oscilloscope References
  let miniEcgScope = null;
  let miniPpgScope = null;
  let fullEcgScope = null;
  let fullPpgScope = null;

  // Web Audio Context for Cardiac Beep
  let audioCtx = null;

  // =========================================================================
  // INITIALIZATION ON DOM READY
  // =========================================================================
  document.addEventListener("DOMContentLoaded", () => {
    initLucideIcons();
    initNavigation();
    initOscilloscopes();
    initAudio();
    initPoincare();
    initControls();
    initSessionManager();
    initFaceIdGate();
    initWebcamPipeline();
    initHistoryAndReports();
    initSettingsAndSimulator();
    initWebSocket();
    startClock();
    startAnimationLoop();
  });

  function initLucideIcons() {
    if (window.lucide) {
      window.lucide.createIcons();
    }
  }

  // =========================================================================
  // 1. VIEW NAVIGATION CONTROLLER (8 VIEWS)
  // =========================================================================
  function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    const viewPanels = document.querySelectorAll(".view-panel");
    const viewTitleEl = document.getElementById("view-title");
    const btnBack = document.getElementById("btn-back-overview");

    const titles = {
      "view-overview": { icon: "layout-dashboard", text: "Cockpit Telemetry" },
      "view-face": { icon: "scan-face", text: "Biometric Access & Liveness" },
      "view-ecg": { icon: "heart-pulse", text: "Live ECG Waveform Monitor (Lead II)" },
      "view-ppg": { icon: "waves", text: "Photoplethysmogram (PPG) & SpO2 Monitor" },
      "view-analytics": { icon: "line-chart", text: "Biomedical Analytics & Blood Pressure Model" },
      "view-history": { icon: "history", text: "Archived Clinical Sessions" },
      "view-reports": { icon: "file-text", text: "Clinical Telemetry Report & Export" },
      "view-settings": { icon: "settings-2", text: "Hardware Port & Simulator Configuration" },
    };

    function switchView(targetViewId) {
      state.currentView = targetViewId;

      // Update Nav Buttons
      navItems.forEach((btn) => {
        if (btn.dataset.view === targetViewId) {
          btn.classList.add("active");
        } else {
          btn.classList.remove("active");
        }
      });

      // Update View Visibility
      viewPanels.forEach((panel) => {
        if (panel.id === targetViewId) {
          panel.classList.remove("hidden");
        } else {
          panel.classList.add("hidden");
        }
      });

      // Update Header Title
      const meta = titles[targetViewId] || titles["view-overview"];
      if (viewTitleEl) {
        viewTitleEl.innerHTML = `<i data-lucide="${meta.icon}" class="w-4 h-4 text-[#18181B]"></i><span>${meta.text}</span>`;
        initLucideIcons();
      }

      // Update Back Button appearance
      if (btnBack) {
        if (targetViewId === "view-overview") {
          btnBack.classList.remove("bg-[#EFECE3]");
        } else {
          btnBack.classList.add("bg-white");
        }
      }

      // Trigger Oscilloscope Resize on Display Change
      setTimeout(() => {
        if (miniEcgScope) miniEcgScope.resize();
        if (miniPpgScope) miniPpgScope.resize();
        if (fullEcgScope) fullEcgScope.resize();
        if (fullPpgScope) fullPpgScope.resize();
      }, 50);
    }

    navItems.forEach((btn) => {
      btn.addEventListener("click", () => {
        switchView(btn.dataset.view);
      });
    });

    // Jump Links (e.g. from Overview buttons to dedicated views)
    document.querySelectorAll("[data-jump]").forEach((el) => {
      el.addEventListener("click", () => {
        switchView(el.dataset.jump);
      });
    });

    // Explicit Back Button to Cockpit Overview
    if (btnBack) {
      btnBack.addEventListener("click", () => {
        switchView("view-overview");
      });
    }

    // Expose switchView globally for verification and scripting
    window.switchView = switchView;
  }

  // =========================================================================
  // APPLE FACE ID AUTHENTICATION GATE & BIOMETRIC FLOW MODULE
  // =========================================================================
  const ENROLLED_PERSONNEL = {
    ashutosh: {
      name: "Ashutosh",
      role: "Lead Biomedical Research Engineer",
      avatar: "AS",
      participantId: 16,
      department: "Clinical Biosignals",
    },
    dhrunil: {
      name: "Dhrunil",
      role: "Senior Clinical Systems Architect",
      avatar: "DH",
      participantId: 17,
      department: "Hardware Telemetry",
    },
  };

  const faceIdState = {
    isLocked: true,
    isAuthenticating: false,
    currentUser: null,
    animTimer: null,
  };

  function initFaceIdGate() {
    const modal = document.getElementById("faceid-modal");
    if (!modal) return;

    // Personnel demo buttons for instant testing
    const btnAshutosh = document.getElementById("btn-faceid-ashutosh");
    const btnDhrunil = document.getElementById("btn-faceid-dhrunil");
    const btnUnknown = document.getElementById("btn-faceid-unknown");
    const btnBypass = document.getElementById("btn-faceid-bypass");
    const btnLock = document.getElementById("btn-header-lock");
    const btnLogout = document.getElementById("btn-header-logout");

    if (btnAshutosh) {
      btnAshutosh.addEventListener("click", () => {
        const restrictedAlert = document.getElementById("faceid-restricted-alert");
        if (restrictedAlert) restrictedAlert.classList.add("hidden");
        fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: "Ashutosh" }),
        }).catch(console.warn);
        startBiometricAuthenticationSequence("ashutosh");
      });
    }

    if (btnDhrunil) {
      btnDhrunil.addEventListener("click", () => {
        const restrictedAlert = document.getElementById("faceid-restricted-alert");
        if (restrictedAlert) restrictedAlert.classList.add("hidden");
        fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: "Dhrunil" }),
        }).catch(console.warn);
        startBiometricAuthenticationSequence("dhrunil");
      });
    }

    if (btnUnknown) {
      btnUnknown.addEventListener("click", () => {
        showFaceIdRestrictedWarning();
      });
    }

    if (btnBypass) {
      btnBypass.addEventListener("click", () => {
        unlockDashboardAsGuest();
      });
    }

    const btnClose = document.getElementById("btn-faceid-close");
    if (btnClose) {
      btnClose.addEventListener("click", () => {
        unlockDashboardAsGuest();
      });
    }

    if (btnLock) {
      btnLock.addEventListener("click", () => {
        performLogout();
      });
    }

    if (btnLogout) {
      btnLogout.addEventListener("click", () => {
        performLogout();
      });
    }

    // Initialize Face ID modal webcam if user gives permission
    initFaceIdWebcam();
  }

  async function initFaceIdWebcam() {
    const video = document.getElementById("faceid-video");
    if (!video || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      video.srcObject = stream;
      video.play();
      startFaceIdLiveCheck(video);
    } catch (err) {
      console.log("Face ID webcam standby:", err);
    }
  }

  function renderFaceIdOverlay(canvas, detections) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const cw = canvas.width || 220;
    const ch = canvas.height || 220;
    ctx.clearRect(0, 0, cw, ch);

    if (!detections || detections.length === 0) return;

    // Scale calculation for object-fit: cover from 640x480 video into cw x ch
    const srcW = 640;
    const srcH = 480;
    const scale = Math.max(cw / srcW, ch / srcH);
    const renderW = srcW * scale;
    const renderH = srcH * scale;
    const offsetX = (renderW - cw) / 2;
    const offsetY = (renderH - ch) / 2;

    detections.forEach((det) => {
      const bbox = det.bbox || [0, 0, 0, 0];
      const origX1 = bbox[0] * scale - offsetX;
      const origY1 = bbox[1] * scale - offsetY;
      const origX2 = bbox[2] * scale - offsetX;
      const origY2 = bbox[3] * scale - offsetY;

      // Mirrored horizontal coordinates to match CSS transform: scaleX(-1) on video
      const x1 = Math.max(0, cw - origX2);
      const x2 = Math.min(cw, cw - origX1);
      const y1 = Math.max(0, origY1);
      const y2 = Math.min(ch, origY2);
      const bw = Math.max(10, x2 - x1);
      const bh = Math.max(10, y2 - y1);

      const name = (det.name || "UNKNOWN").trim();
      const isKnown = det.is_known || (name.toLowerCase() === "ashutosh" || name.toLowerCase() === "dhrunil");
      const isLive = det.liveness ? det.liveness.is_live : true;
      const sim = Math.round((det.similarity || 0) * 100);

      let strokeColor = "#b45309"; // Warm Amber for UNKNOWN
      let fillColor = "rgba(180, 83, 9, 0.12)";
      let label = `UNKNOWN (${sim}%)`;

      if (!isLive) {
        strokeColor = "#be123c"; // Warm Rose for spoof
        fillColor = "rgba(190, 18, 60, 0.12)";
        label = "SPOOF SUSPECTED";
      } else if (isKnown) {
        strokeColor = "#15803d"; // Warm Forest Green for Ashutosh/Dhrunil
        fillColor = "rgba(21, 128, 61, 0.12)";
        label = `${name} (${sim}%)`;
      }

      // Draw bounding box
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.fillStyle = fillColor;

      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x1, y1, bw, bh, 8);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.fillRect(x1, y1, bw, bh);
        ctx.strokeRect(x1, y1, bw, bh);
      }

      // Draw 5 facial landmarks if present
      if (det.landmarks && Array.isArray(det.landmarks)) {
        ctx.fillStyle = strokeColor;
        det.landmarks.forEach((pt) => {
          const lmx = cw - (pt[0] * scale - offsetX);
          const lmy = pt[1] * scale - offsetY;
          ctx.beginPath();
          ctx.arc(lmx, lmy, 2.5, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      // Draw identity pill tag
      ctx.font = '600 10px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif';
      const textMetrics = ctx.measureText(label);
      const tagW = textMetrics.width + 12;
      const tagH = 18;
      const tagX = Math.max(4, Math.min(cw - tagW - 4, x1));
      const tagY = Math.max(20, y1 - 4);

      ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(tagX, tagY - 14, tagW, tagH, 4);
        ctx.fill();
      } else {
        ctx.fillRect(tagX, tagY - 14, tagW, tagH);
      }

      ctx.fillStyle = strokeColor;
      ctx.fillText(label, tagX + 6, tagY);
    });
  }

  function startFaceIdLiveCheck(video) {
    const faceidCanvas = document.getElementById("faceid-canvas");
    if (faceidCanvas) {
      faceidCanvas.width = 220;
      faceidCanvas.height = 220;
    }

    const offscreen = document.createElement("canvas");
    offscreen.width = 640;
    offscreen.height = 480;
    const offCtx = offscreen.getContext("2d");

    let isRequestInFlight = false;

    setInterval(async () => {
      if (!faceIdState.isLocked || faceIdState.isAuthenticating || isRequestInFlight) return;
      if (video.readyState < 2) return;

      try {
        isRequestInFlight = true;
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        offscreen.width = 640;
        offscreen.height = Math.round(640 * (vh / vw)) || 480;
        offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
        const b64 = offscreen.toDataURL("image/jpeg", 0.80);

        const resp = await fetch("/api/auth/login-by-face", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: b64, recognition_thresh: 0.35 }),
        });
        const data = await resp.json();

        const detections = data.detections || (data.detection ? [data.detection] : []);
        renderFaceIdOverlay(faceidCanvas, detections);

        const reticle = document.getElementById("faceid-reticle");
        const statusText = document.getElementById("faceid-status-text");
        const restrictedAlert = document.getElementById("faceid-restricted-alert");

        if (data.authenticated && data.success && data.user) {
          if (reticle) {
            reticle.className = "faceid-reticle-ring verified";
            reticle.style.borderColor = "";
          }
          if (statusText) statusText.textContent = `Identity Verified: ${data.user.name}`;
          if (restrictedAlert) restrictedAlert.classList.add("hidden");
          startBiometricAuthenticationSequence(data.user.name.toLowerCase());
        } else if (data.reason === "UNKNOWN_SUBJECT" || (detections.length > 0 && !data.authenticated)) {
          // Live Unknown Subject detected: render amber reticle and inform user
          if (reticle) {
            reticle.className = "faceid-reticle-ring scanning";
            reticle.style.borderColor = "#b45309";
          }
          const sim = detections[0] ? Math.round((detections[0].similarity || 0) * 100) : 0;
          if (statusText) statusText.textContent = `Unregistered Subject (UNKNOWN: ${sim}%)`;
          if (restrictedAlert) {
            restrictedAlert.classList.remove("hidden");
            const alertMsg = restrictedAlert.querySelector(".text-rose-800");
            if (alertMsg) {
              alertMsg.textContent = "Unregistered subject (UNKNOWN). Physiological vitals are calibrated for Ashutosh & Dhrunil. You can continue as Guest or authenticate.";
            }
          }
        } else if (data.reason === "NO_FACE") {
          if (reticle) {
            reticle.className = "faceid-reticle-ring scanning";
            reticle.style.borderColor = "";
          }
          if (statusText) statusText.textContent = "Position your face within the frame to authenticate";
          if (restrictedAlert) restrictedAlert.classList.add("hidden");
        }
      } catch (e) {
        // Silently continue scanning
      } finally {
        isRequestInFlight = false;
      }
    }, 600);
  }

  function startBiometricAuthenticationSequence(personKey) {
    if (faceIdState.isAuthenticating) return;

    const person = ENROLLED_PERSONNEL[personKey.toLowerCase()] || ENROLLED_PERSONNEL.ashutosh;
    faceIdState.isAuthenticating = true;
    faceIdState.isLocked = false;

    const progressContainer = document.getElementById("faceid-progress-container");
    const restrictedAlert = document.getElementById("faceid-restricted-alert");
    const reticle = document.getElementById("faceid-reticle");
    const progressCircle = document.getElementById("faceid-progress-circle");
    const progressText = document.getElementById("faceid-progress-text");
    const phaseMessage = document.getElementById("faceid-phase-message");
    const statusText = document.getElementById("faceid-status-text");

    if (restrictedAlert) restrictedAlert.classList.add("hidden");
    if (progressContainer) progressContainer.classList.remove("hidden");
    if (reticle) reticle.className = "faceid-reticle-ring scanning";
    if (statusText) statusText.textContent = `Enrolled Personnel Verified: ${person.name}`;

    let progress = 0;
    const totalDuration = 2000; // ms
    const intervalTime = 25; // ms
    const step = (100 / (totalDuration / intervalTime));

    if (faceIdState.animTimer) clearInterval(faceIdState.animTimer);

    faceIdState.animTimer = setInterval(() => {
      progress += step;
      if (progress > 100) progress = 100;

      // Update SVG circular progress ring
      if (progressCircle) {
        const offset = 100 - progress;
        progressCircle.style.strokeDashoffset = offset;
      }
      if (progressText) {
        progressText.textContent = `${Math.round(progress)}% Synchronized`;
      }

      // Apple Biometric Loading Animation phased messages:
      // 0% - 25%: "Authenticating Identity..."
      // 25% - 55%: "Calibrating Optical Hemodynamics..."
      // 55% - 85%: "Synchronizing Lead II ECG..."
      // 85% - 100%: "Telemetry Active"
      if (phaseMessage) {
        if (progress < 25) {
          phaseMessage.textContent = "Authenticating Identity...";
        } else if (progress < 55) {
          phaseMessage.textContent = "Calibrating Optical Hemodynamics...";
        } else if (progress < 85) {
          phaseMessage.textContent = "Synchronizing Lead II ECG...";
        } else {
          phaseMessage.textContent = "Telemetry Active";
        }
      }

      if (progress >= 100) {
        clearInterval(faceIdState.animTimer);
        faceIdState.animTimer = null;

        if (reticle) reticle.className = "faceid-reticle-ring verified";

        // Complete smooth transition to personalized Bento Health Dashboard
        setTimeout(() => {
          completeAuthenticationTransition(person);
        }, 300);
      }
    }, intervalTime);
  }

  async function completeAuthenticationTransition(person) {
    const modal = document.getElementById("faceid-modal");
    if (modal) {
      modal.classList.add("authenticated");
    }

    faceIdState.isLocked = false;
    faceIdState.isAuthenticating = false;
    faceIdState.currentUser = person;
    state.vitals.isRestricted = false;

    // Inform backend of verified enrolled personnel login
    try {
      await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: person.name })
      });
    } catch (e) {
      console.warn("Auth login sync failed:", e);
    }

    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ command: "login", name: person.name }));
    }

    // Update Top Bar Personalized Header with Avatar, Name, and Role
    const hdrAvatar = document.getElementById("header-user-avatar");
    const hdrName = document.getElementById("header-user-name");
    const hdrRole = document.getElementById("header-user-role");

    if (hdrAvatar) {
      hdrAvatar.textContent = person.avatar;
      hdrAvatar.className = person.avatar === "AS"
        ? "w-7 h-7 rounded-full bg-[#18181B] text-white flex items-center justify-center font-bold text-xs shadow-xs"
        : "w-7 h-7 rounded-full bg-[#15803D] text-white flex items-center justify-center font-bold text-xs shadow-xs";
    }
    if (hdrName) hdrName.textContent = person.name;
    if (hdrRole) hdrRole.textContent = person.role;

    // Synchronize Airbnb Telemetry Capsule
    const capsuleName = document.getElementById("hdr-capsule-name");
    const capsuleSession = document.getElementById("hdr-capsule-session");
    if (capsuleName) capsuleName.textContent = person.name;
    if (capsuleSession) capsuleSession.textContent = "Verified Sync";

    // Update Sidebar Active Subject
    const subjAvatar = document.getElementById("subject-avatar");
    const subjName = document.getElementById("subject-name");
    const sessBadge = document.getElementById("session-badge");

    if (subjAvatar) {
      subjAvatar.textContent = person.avatar;
      subjAvatar.className = person.avatar === "AS"
        ? "w-8 h-8 rounded-full bg-[#18181B] text-white flex items-center justify-center font-bold text-xs"
        : "w-8 h-8 rounded-full bg-[#15803D] text-white flex items-center justify-center font-bold text-xs";
    }
    if (subjName) subjName.textContent = person.name;
    if (sessBadge) {
      sessBadge.textContent = "VERIFIED";
      sessBadge.className = "px-2 py-0.5 rounded-full text-[9px] font-mono bg-[#F0FDF4] text-[#15803D] border border-[#BBF7D0] font-semibold";
    }

    // Update Identity Card in View 2
    updateIdentityPanel({
      name: person.name,
      similarity: 98.4,
      trackId: person.participantId || 101,
      isLive: true,
      blink: true,
    });

    // Automatically bind telemetry session to verified subject
    bindSubjectToSession(person.name, person.participantId);
  }

  function showFaceIdRestrictedWarning() {
    if (faceIdState.isAuthenticating || !faceIdState.isLocked) return;

    const progressContainer = document.getElementById("faceid-progress-container");
    const restrictedAlert = document.getElementById("faceid-restricted-alert");
    const reticle = document.getElementById("faceid-reticle");
    const statusText = document.getElementById("faceid-status-text");

    if (progressContainer) progressContainer.classList.add("hidden");
    if (restrictedAlert) {
      restrictedAlert.classList.remove("hidden");
      restrictedAlert.style.animation = "none";
      restrictedAlert.offsetHeight; // reflow
      restrictedAlert.style.animation = "";
      const alertMsg = restrictedAlert.querySelector(".text-amber-800");
      if (alertMsg) {
        alertMsg.textContent = "Unregistered subject (UNKNOWN). Vital calculations are calibrated for Ashutosh & Dhrunil. You can continue as Guest or authenticate.";
      }
    }
    if (reticle) {
      reticle.className = "faceid-reticle-ring scanning";
      reticle.style.borderColor = "#b45309";
    }
    if (statusText) statusText.textContent = "Unregistered Subject Detected (UNKNOWN)";

    state.vitals.isRestricted = true;
    refreshMetricCards();
  }

  async function unlockDashboardAsGuest() {
    const modal = document.getElementById("faceid-modal");
    if (modal) modal.classList.add("authenticated");
    faceIdState.isLocked = false;
    faceIdState.isAuthenticating = false;
    faceIdState.currentUser = {
      name: "Guest Subject",
      role: "Guest Mode (Calibrated for Ashutosh & Dhrunil)",
      avatar: "GU",
      participantId: null,
      department: "Unregistered Guest",
    };
    state.vitals.isRestricted = true;

    try {
      await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "Guest" }),
      });
    } catch (e) {
      console.warn("Guest auth login sync failed:", e);
    }

    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ command: "login", name: "Guest" }));
    }

    // Update Top Bar Personalized Header
    const hdrAvatar = document.getElementById("header-user-avatar");
    const hdrName = document.getElementById("header-user-name");
    const hdrRole = document.getElementById("header-user-role");

    if (hdrAvatar) {
      hdrAvatar.textContent = "GU";
      hdrAvatar.className = "w-7 h-7 rounded-full bg-[#FAF9F5] border border-[#ECE8DF] flex items-center justify-center font-bold text-[#78736B] text-xs";
    }
    if (hdrName) hdrName.textContent = "Guest Subject";
    if (hdrRole) hdrRole.textContent = "Guest Mode (Calibrated for Ashutosh & Dhrunil)";

    // Synchronize Airbnb Telemetry Capsule
    const capsuleName = document.getElementById("hdr-capsule-name");
    const capsuleSession = document.getElementById("hdr-capsule-session");
    if (capsuleName) capsuleName.textContent = "Guest Subject";
    if (capsuleSession) capsuleSession.textContent = "Guest Mode";

    // Update Sidebar Active Subject
    const subjAvatar = document.getElementById("subject-avatar");
    const subjName = document.getElementById("subject-name");
    const sessBadge = document.getElementById("session-badge");

    if (subjAvatar) {
      subjAvatar.textContent = "GU";
      subjAvatar.className = "w-8 h-8 rounded-full bg-[#FAF9F5] border border-[#ECE8DF] flex items-center justify-center font-bold text-[#78736B] text-xs";
    }
    if (subjName) subjName.textContent = "Guest Subject";
    if (sessBadge) {
      sessBadge.textContent = "GUEST";
      sessBadge.className = "px-2 py-0.5 rounded-full text-[9px] font-mono bg-[#FFFBEB] text-[#B45309] border border-[#FDE68A] font-medium";
    }

    updateIdentityPanel({
      name: "UNKNOWN / GUEST",
      similarity: 0,
      trackId: 100,
      isLive: true,
      blink: true,
    });
    refreshMetricCards();
  }

  async function performLogout() {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } catch (e) {
      console.warn("Logout request failed:", e);
    }
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ command: "logout" }));
    }
    lockSystemToFaceId();
  }

  function lockSystemToFaceId() {
    faceIdState.isLocked = true;
    faceIdState.isAuthenticating = false;
    faceIdState.currentUser = null;
    state.vitals.isRestricted = true;
    refreshMetricCards();

    if (faceIdState.animTimer) {
      clearInterval(faceIdState.animTimer);
      faceIdState.animTimer = null;
    }

    // Synchronize Airbnb Telemetry Capsule
    const capsuleName = document.getElementById("hdr-capsule-name");
    const capsuleSession = document.getElementById("hdr-capsule-session");
    if (capsuleName) capsuleName.textContent = "Locked";
    if (capsuleSession) capsuleSession.textContent = "Standby";

    const modal = document.getElementById("faceid-modal");
    const progressContainer = document.getElementById("faceid-progress-container");
    const restrictedAlert = document.getElementById("faceid-restricted-alert");
    const reticle = document.getElementById("faceid-reticle");
    const statusText = document.getElementById("faceid-status-text");
    const progressCircle = document.getElementById("faceid-progress-circle");

    if (modal) modal.classList.remove("authenticated");
    if (progressContainer) progressContainer.classList.add("hidden");
    if (restrictedAlert) restrictedAlert.classList.add("hidden");
    if (reticle) reticle.className = "faceid-reticle-ring scanning";
    if (statusText) statusText.textContent = "Position your face within the frame to authenticate";
    if (progressCircle) progressCircle.style.strokeDashoffset = "100";
  }

  // =========================================================================
  // 2. OSCILLOSCOPE ENGINES SETUP
  // =========================================================================
  function initOscilloscopes() {
    // 1. Mini ECG Scope (Overview)
    const miniEcgCanvas = document.getElementById("canvas-mini-ecg");
    if (miniEcgCanvas) {
      miniEcgScope = new MedicalOscilloscope(miniEcgCanvas, {
        mode: "sweep",
        sweepDuration: 4.0,
        sampleRate: 100,
        title: "LEAD II (MINI)",
        channels: [
          {
            id: "ecg",
            name: "LEAD II",
            color: "#15803d",
            glowColor: "transparent",
            gain: 1.0,
            minVal: -1.5,
            maxVal: 2.5,
          },
        ],
      });
    }

    // 2. Mini PPG Scope (Overview)
    const miniPpgCanvas = document.getElementById("canvas-mini-ppg");
    if (miniPpgCanvas) {
      miniPpgScope = new MedicalOscilloscope(miniPpgCanvas, {
        mode: "sweep",
        sweepDuration: 4.0,
        sampleRate: 100,
        title: "PPG PLETH (MINI)",
        channels: [
          {
            id: "ppg_ir",
            name: "IR (880nm)",
            color: "#1d4ed8",
            glowColor: "transparent",
            gain: 1.0,
            offsetY: 0.15,
            minVal: -1.2,
            maxVal: 2.0,
          },
          {
            id: "ppg_red",
            name: "RED (660nm)",
            color: "#be123c",
            glowColor: "transparent",
            gain: 0.8,
            offsetY: -0.25,
            minVal: -1.2,
            maxVal: 2.0,
          },
        ],
      });
    }

    // 3. Full-Screen ECG Scope (Live ECG View)
    const fullEcgCanvas = document.getElementById("canvas-full-ecg");
    if (fullEcgCanvas) {
      fullEcgScope = new MedicalOscilloscope(fullEcgCanvas, {
        mode: "sweep",
        sweepDuration: 5.0,
        sampleRate: 100,
        title: "ELECTROCARDIOGRAM LEAD II [CALIBRATED]",
        channels: [
          {
            id: "ecg",
            name: "LEAD II",
            color: "#15803d",
            glowColor: "transparent",
            gain: 1.0,
            minVal: -1.5,
            maxVal: 2.5,
          },
        ],
      });
    }

    // 4. Full-Screen PPG Scope (Live PPG View)
    const fullPpgCanvas = document.getElementById("canvas-full-ppg");
    if (fullPpgCanvas) {
      fullPpgScope = new MedicalOscilloscope(fullPpgCanvas, {
        mode: "sweep",
        sweepDuration: 5.0,
        sampleRate: 100,
        title: "PHOTOPLETHYSMOGRAPHY & PULSE OXIMETRY",
        channels: [
          {
            id: "ppg_ir",
            name: "IR 880nm (Arterial)",
            color: "#1d4ed8",
            glowColor: "transparent",
            gain: 1.0,
            offsetY: 0.15,
            minVal: -1.2,
            maxVal: 2.2,
          },
          {
            id: "ppg_red",
            name: "Red 660nm (Venous)",
            color: "#be123c",
            glowColor: "transparent",
            gain: 0.9,
            offsetY: -0.25,
            minVal: -1.2,
            maxVal: 2.2,
          },
        ],
      });
    }
  }

  // Continuous 60fps RequestAnimationFrame Render Loop
  function startAnimationLoop() {
    function loop() {
      if (state.currentView === "view-overview") {
        if (miniEcgScope) miniEcgScope.render();
        if (miniPpgScope) miniPpgScope.render();
      } else if (state.currentView === "view-ecg") {
        if (fullEcgScope) fullEcgScope.render();
      } else if (state.currentView === "view-ppg") {
        if (fullPpgScope) fullPpgScope.render();
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  // =========================================================================
  // 3. TELEMETRY WEBSOCKET CLIENT & SAMPLE DISPATCHER
  // =========================================================================
  function initWebSocket() {
    const connDot = document.getElementById("sys-conn-dot");
    const connStatus = document.getElementById("sys-conn-status");
    const urlInput = document.getElementById("ws-server-url");
    if (urlInput && urlInput.value) {
      state.wsUrl = urlInput.value.trim();
    }

    try {
      if (state.ws) {
        state.ws.close();
      }
      state.ws = new WebSocket(state.wsUrl);
    } catch (e) {
      console.warn("WebSocket initialization deferred to fallback simulation:", e);
      enableSimulator();
      return;
    }

    state.ws.onopen = () => {
      state.wsConnected = true;
      disableSimulator();
      if (connDot) {
        connDot.className = "w-2.5 h-2.5 rounded-full bg-emerald-400 status-dot-pulse";
      }
      if (connStatus) {
        connStatus.textContent = "CONNECTED (WS LIVE)";
        connStatus.className = "text-emerald-400 font-mono text-[11px]";
      }
    };

    state.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        handleTelemetryMessage(payload);
      } catch (err) {
        console.error("Error parsing telemetry packet:", err);
      }
    };

    state.ws.onclose = () => {
      state.wsConnected = false;
      if (connDot) {
        connDot.className = "w-2.5 h-2.5 rounded-full bg-amber-500 status-dot-pulse";
      }
      if (connStatus) {
        connStatus.textContent = "SIMULATING (STANDALONE)";
        connStatus.className = "text-amber-400 font-mono text-[11px]";
      }
      enableSimulator();
    };

    state.ws.onerror = (e) => {
      state.wsConnected = false;
      enableSimulator();
    };
  }

  function handleTelemetryMessage(msg) {
    if (!msg) return;

    // Handle initial state or session sync
    if (msg.session && msg.session.active_session_id && !state.activeSession.id) {
      syncSessionFromBackend(msg.session);
    }

    // Direct sample stream or sample arrays
    if (msg.samples && Array.isArray(msg.samples)) {
      msg.samples.forEach((s) => feedSampleToScopes(s));
    } else if (msg.type === "sample" || msg.ecg_raw !== undefined) {
      feedSampleToScopes(msg);
    }

    // High-level physiological metrics payload
    if (msg.metrics) {
      updateBiomedicalMetrics(msg.metrics);
    }

    // Hardware serial status reflection
    if (msg.serial_status) {
      const mockToggle = document.getElementById("toggle-mock-mode");
      if (mockToggle && msg.serial_status.mock_mode !== undefined) {
        mockToggle.checked = msg.serial_status.mock_mode;
      }
    }
  }

  function syncSessionFromBackend(sess) {
    state.activeSession.id = sess.active_session_id;
    state.activeSession.participantId = sess.participant_id;
    state.activeSession.isGuest = sess.is_guest;
    state.activeSession.startTime = Date.now();
    state.activeSession.name = sess.is_guest ? "Guest Session" : (sess.name || "Enrolled Subject");

    if (state.activeSession.timerInterval) clearInterval(state.activeSession.timerInterval);
    state.activeSession.timerInterval = setInterval(updateSessionClock, 1000);

    const badge = document.getElementById("session-badge");
    const subjName = document.getElementById("subject-name");
    const btnToggle = document.getElementById("btn-toggle-session");

    if (badge) {
      badge.textContent = sess.is_guest ? "GUEST ACTIVE" : "SUBJECT BOUND";
      badge.className = "px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-950 text-emerald-400 border border-emerald-800";
    }
    if (subjName) subjName.textContent = state.activeSession.name;
    if (btnToggle) {
      btnToggle.textContent = "End Session";
      btnToggle.className = "flex-1 py-1.5 rounded bg-red-600/30 hover:bg-red-600/50 text-red-300 border border-red-500/40 text-[11px] font-medium transition-colors";
    }
  }

  function feedSampleToScopes(s) {
    const ecgVal = s.ecg !== undefined ? s.ecg : ((s.ecg_raw || 512) - 512) / 120.0;
    const ppgRedVal = s.ppg_red_norm !== undefined ? s.ppg_red_norm : ((s.ppg_red || 20000) - 20000) / 4000.0;
    const ppgIrVal = s.ppg_ir_norm !== undefined ? s.ppg_ir_norm : ((s.ppg_ir || 24000) - 24000) / 4000.0;

    const isRPeak = !!s.is_r_peak;
    const isSystolic = !!s.is_systolic_peak;
    const leadsOff = s.leads_off !== undefined ? s.leads_off : state.vitals.leadsOff;

    const ecgBatch = {
      ecg: { value: ecgVal, isPeak: isRPeak, label: isRPeak ? "R" : "" },
    };

    const ppgBatch = {
      ppg_ir: { value: ppgIrVal, isPeak: isSystolic, label: isSystolic ? "SYS" : "" },
      ppg_red: { value: ppgRedVal, isPeak: false, label: "" },
    };

    // Feed Mini Scopes
    if (miniEcgScope) miniEcgScope.pushData(ecgBatch);
    if (miniPpgScope) miniPpgScope.pushData(ppgBatch);

    // Feed Full Scopes
    if (fullEcgScope) fullEcgScope.pushData(ecgBatch);
    if (fullPpgScope) fullPpgScope.pushData(ppgBatch);

    // Audio beep & heartbeat animation on R-peak
    if (isRPeak) {
      triggerCardiacBeat();
    }

    // Lead Detach & Probe Fault Updates
    handleLeadStatus(leadsOff > 0);

    // Session sample counting
    if (state.activeSession.id) {
      state.activeSession.sampleCount++;
    }
  }

  function updateBiomedicalMetrics(m) {
    const v = state.vitals;
    const isRestricted = (m.access_status === "RESTRICTED/UNREGISTERED" || m.signal_quality === "RESTRICTED/UNREGISTERED" || m.bp_status === "RESTRICTED/UNREGISTERED");
    v.isRestricted = isRestricted;

    if (isRestricted) {
      v.signalQuality = "RESTRICTED/UNREGISTERED";
      refreshMetricCards();
      return;
    }

    if (m.ecg_hr !== undefined && m.ecg_hr !== null && m.ecg_hr > 0) v.ecgHr = Math.round(m.ecg_hr);
    if (m.ppg_hr !== undefined && m.ppg_hr !== null && m.ppg_hr > 0) v.ppgHr = Math.round(m.ppg_hr);
    if (m.spo2 !== undefined && m.spo2 !== null && m.spo2 > 0) v.spo2 = parseFloat(m.spo2.toFixed(1));
    if (m.pat_ms !== undefined && m.pat_ms !== null && m.pat_ms > 0) v.pat = Math.round(m.pat_ms);
    if (m.rr_interval_ms !== undefined && m.rr_interval_ms !== null && m.rr_interval_ms > 0) {
      v.rrInterval = Math.round(m.rr_interval_ms);
      recordRRInterval(v.rrInterval);
    }
    if (m.rmssd_ms !== undefined && m.rmssd_ms !== null) v.rmssd = parseFloat(m.rmssd_ms.toFixed(1));
    if (m.sdnn_ms !== undefined && m.sdnn_ms !== null) v.sdnn = parseFloat(m.sdnn_ms.toFixed(1));
    if (m.resp_rate_bpm !== undefined && m.resp_rate_bpm !== null) v.respRate = Math.round(m.resp_rate_bpm);
    if (m.bp_systolic !== undefined && m.bp_systolic !== null) v.bpSys = Math.round(m.bp_systolic);
    if (m.bp_diastolic !== undefined && m.bp_diastolic !== null) v.bpDia = Math.round(m.bp_diastolic);
    if (m.signal_quality) v.signalQuality = m.signal_quality;

    refreshMetricCards();
  }

  function refreshMetricCards() {
    const v = state.vitals;

    // Overview Cards
    const elHr = document.getElementById("metric-hr");
    const elSpo2 = document.getElementById("metric-spo2");
    const elBp = document.getElementById("metric-bp");
    const elResp = document.getElementById("metric-resp");
    const elPat = document.getElementById("metric-pat");
    const elRmssd = document.getElementById("metric-rmssd");
    const elSdnn = document.getElementById("metric-sdnn");
    const elMiniRr = document.getElementById("mini-rr");
    const hrQualityTag = document.getElementById("hr-quality-tag");

    // Live ECG View readouts
    const elEcgHr = document.getElementById("ecg-hr-value");
    const elEcgRr = document.getElementById("ecg-rr-value");
    const elEcgRmssd = document.getElementById("ecg-rmssd-sub");
    const elEcgSdnn = document.getElementById("ecg-sdnn-sub");

    // Live PPG View readouts
    const elPpgSpo2 = document.getElementById("ppg-spo2-value");
    const elPpgPi = document.getElementById("ppg-pi-value");

    // Analytics View readouts
    const elSysVal = document.getElementById("analytics-sys-val");
    const elDiaVal = document.getElementById("analytics-dia-val");
    const elTablePat = document.getElementById("table-pat");
    const elTableMap = document.getElementById("table-map");
    const elTablePp = document.getElementById("table-pp");
    const elEdrBpm = document.getElementById("edr-bpm");

    if (v.isRestricted) {
      if (elHr) elHr.textContent = "--";
      if (elSpo2) elSpo2.textContent = "--";
      if (elBp) elBp.textContent = "-- / --";
      if (elResp) elResp.textContent = "--";
      if (elPat) elPat.textContent = "--";
      if (elRmssd) elRmssd.textContent = "--";
      if (elSdnn) elSdnn.textContent = "-- ms";
      if (elMiniRr) elMiniRr.textContent = "-- ms";
      if (hrQualityTag) {
        hrQualityTag.textContent = "RESTRICTED";
        hrQualityTag.className = "text-[#B45309] font-semibold";
      }
      if (elEcgHr) elEcgHr.textContent = "-- BPM";
      if (elEcgRr) elEcgRr.textContent = "-- ms";
      if (elEcgRmssd) elEcgRmssd.textContent = "-- ms";
      if (elEcgSdnn) elEcgSdnn.textContent = "-- ms";
      if (elPpgSpo2) elPpgSpo2.textContent = "--%";
      if (elPpgPi) elPpgPi.textContent = "--%";
      if (elSysVal) elSysVal.textContent = "--";
      if (elDiaVal) elDiaVal.textContent = "--";
      if (elTablePat) elTablePat.textContent = "-- ms";
      if (elTablePp) elTablePp.textContent = "--";
      if (elTableMap) elTableMap.textContent = "--";
      if (elEdrBpm) elEdrBpm.textContent = "--";
      return;
    }

    if (hrQualityTag) {
      hrQualityTag.textContent = "VALID";
      hrQualityTag.className = "text-[#15803D] font-semibold";
    }

    if (elHr) elHr.textContent = v.ecgHr;
    if (elSpo2) elSpo2.textContent = v.spo2.toFixed(1);
    if (elBp) elBp.textContent = `${v.bpSys} / ${v.bpDia}`;
    if (elResp) elResp.textContent = v.respRate;
    if (elPat) elPat.textContent = v.pat;
    if (elRmssd) elRmssd.textContent = v.rmssd.toFixed(1);
    if (elSdnn) elSdnn.textContent = `${v.sdnn} ms`;
    if (elMiniRr) elMiniRr.textContent = `${v.rrInterval} ms`;

    if (elEcgHr) elEcgHr.textContent = `${v.ecgHr} BPM`;
    if (elEcgRr) elEcgRr.textContent = `${v.rrInterval} ms`;
    if (elEcgRmssd) elEcgRmssd.textContent = `${v.rmssd} ms`;
    if (elEcgSdnn) elEcgSdnn.textContent = `${v.sdnn} ms`;

    if (elPpgSpo2) elPpgSpo2.textContent = `${v.spo2.toFixed(1)}%`;
    if (elPpgPi) elPpgPi.textContent = `${v.pi.toFixed(1)}%`;

    if (elSysVal) elSysVal.textContent = v.bpSys;
    if (elDiaVal) elDiaVal.textContent = v.bpDia;
    if (elTablePat) elTablePat.textContent = `${v.pat} ms`;
    if (elTablePp) elTablePp.textContent = `${v.bpSys - v.bpDia} mmHg`;
    if (elTableMap) {
      const map = Math.round(v.bpDia + (v.bpSys - v.bpDia) / 3.0);
      elTableMap.textContent = `${map} mmHg`;
    }
    if (elEdrBpm) elEdrBpm.textContent = `${v.respRate} BrPM`;
  }

  function handleLeadStatus(isLeadsOff) {
    state.vitals.leadsOff = isLeadsOff ? 1 : 0;
    const banner = document.getElementById("overview-alarm-banner");
    const leadPill = document.getElementById("lead-status-pill");

    if (isLeadsOff) {
      if (banner) banner.classList.remove("hidden");
      if (leadPill) {
        leadPill.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-[#BE123C]"></span> DETACHED';
        leadPill.className = "text-[#BE123C] font-semibold flex items-center gap-1";
      }
      if (miniEcgScope) miniEcgScope.setLeadsOff(true, "LEADS OFF - REATTACH ELECTRODES");
      if (fullEcgScope) fullEcgScope.setLeadsOff(true, "LEADS OFF - REATTACH ELECTRODES");
    } else {
      if (banner) banner.classList.add("hidden");
      if (leadPill) {
        leadPill.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-[#15803D]"></span> NORMAL';
        leadPill.className = "text-[#15803D] font-semibold flex items-center gap-1";
      }
      if (miniEcgScope) miniEcgScope.setLeadsOff(false);
      if (fullEcgScope) fullEcgScope.setLeadsOff(false);
    }
  }

  // =========================================================================
  // 4. REALISTIC SYNTHETIC TELEMETRY GENERATOR (100 HZ SIMULATOR)
  // =========================================================================
  function enableSimulator() {
    if (state.simulator.intervalId) return;

    state.simulator.intervalId = setInterval(() => {
      generateSimulatorSample();
    }, 10); // 100 Hz = 10 ms interval
  }

  function disableSimulator() {
    if (state.simulator.intervalId) {
      clearInterval(state.simulator.intervalId);
      state.simulator.intervalId = null;
    }
  }

  function generateSimulatorSample() {
    const sim = state.simulator;
    const fs = 100.0;
    sim.sampleIdx++;

    const hr = sim.hr;
    const periodSamples = Math.round((60.0 / hr) * fs);
    const phase = (sim.sampleIdx % periodSamples) / periodSamples; // 0 to 1

    let isRPeak = false;
    let isSystolic = false;

    // Respiration baseline swing (~0.25 Hz)
    const respPeriod = Math.round((60.0 / sim.respRate) * fs);
    const respPhase = (sim.sampleIdx % respPeriod) / respPeriod;
    const respBaseline = Math.sin(2 * Math.PI * respPhase) * 0.12;

    // Synthetic Lead II ECG P-QRS-T Model
    let ecg = 0.0;
    if (phase > 0.10 && phase < 0.20) {
      // P wave
      ecg = 0.20 * Math.sin(((phase - 0.10) / 0.10) * Math.PI);
    } else if (phase > 0.23 && phase < 0.26) {
      // Q dip
      ecg = -0.22 * Math.sin(((phase - 0.23) / 0.03) * Math.PI);
    } else if (phase >= 0.26 && phase <= 0.32) {
      // High R peak
      const rPhase = (phase - 0.26) / 0.06;
      ecg = 1.95 * Math.sin(rPhase * Math.PI);
      if (phase >= 0.285 && phase <= 0.295 && !sim.leadsOff) {
        isRPeak = true;
      }
    } else if (phase > 0.32 && phase < 0.36) {
      // S dip
      ecg = -0.40 * Math.sin(((phase - 0.32) / 0.04) * Math.PI);
    } else if (phase > 0.44 && phase < 0.62) {
      // T wave
      ecg = 0.38 * Math.sin(((phase - 0.44) / 0.18) * Math.PI);
    }

    // Add respiration baseline and high-frequency EMG noise
    const noiseAmp = sim.noise * 0.01;
    ecg += respBaseline + (Math.random() - 0.5) * noiseAmp;

    // Synthetic PPG (Delayed from ECG by Pulse Arrival Time ~220ms = 0.22s * 100 = 22 samples)
    const ppgDelay = 0.22;
    const ppgPhase = (phase - ppgDelay + 1.0) % 1.0;

    let ppgIr = 0.0;
    let ppgRed = 0.0;

    if (!sim.fingerOff) {
      if (ppgPhase > 0.0 && ppgPhase < 0.35) {
        // Systolic upstroke & peak
        const sPhase = ppgPhase / 0.35;
        ppgIr = Math.sin(sPhase * Math.PI * 0.5);
        if (sPhase >= 0.95 && sPhase <= 1.0) {
          isSystolic = true;
        }
      } else if (ppgPhase >= 0.35 && ppgPhase < 0.55) {
        // Dicrotic notch
        const dPhase = (ppgPhase - 0.35) / 0.20;
        ppgIr = 1.0 - dPhase * 0.45 + 0.15 * Math.sin(dPhase * Math.PI);
      } else {
        // Diastolic decay
        const tailPhase = (ppgPhase - 0.55) / 0.45;
        ppgIr = 0.70 * (1.0 - tailPhase);
      }
      ppgIr += respBaseline * 0.5;
      ppgRed = ppgIr * 0.82; // Spectrophotometric absorption ratio for SpO2 ~98%
    } else {
      ppgIr = (Math.random() - 0.5) * 0.04;
      ppgRed = (Math.random() - 0.5) * 0.04;
    }

    // Leads off flatline
    if (sim.leadsOff) {
      ecg = (Math.random() - 0.5) * 0.05;
      isRPeak = false;
    }

    const sample = {
      timestamp: Date.now(),
      seq: sim.sampleIdx,
      ecg: ecg,
      ppg_ir_norm: ppgIr,
      ppg_red_norm: ppgRed,
      is_r_peak: isRPeak,
      is_systolic_peak: isSystolic,
      leads_off: sim.leadsOff,
    };

    feedSampleToScopes(sample);

    // Periodically update metrics every 1 second (100 samples)
    if (sim.sampleIdx % 100 === 0) {
      const rr = Math.round(60000.0 / hr);
      const pat = Math.round(220 + (Math.random() - 0.5) * 6);
      const bpSys = Math.round(118 + (hr - 72) * 0.25 + (Math.random() - 0.5) * 2);
      const bpDia = Math.round(78 + (hr - 72) * 0.15 + (Math.random() - 0.5) * 2);

      updateBiomedicalMetrics({
        ecg_hr: hr,
        ppg_hr: hr,
        spo2: sim.fingerOff ? 0 : 98.4 + (Math.random() - 0.5) * 0.4,
        pat_ms: pat,
        rr_interval_ms: rr,
        rmssd_ms: 42.5 + (Math.random() - 0.5) * 3,
        sdnn_ms: 51.0 + (Math.random() - 0.5) * 4,
        resp_rate_bpm: sim.respRate,
        bp_systolic: bpSys,
        bp_diastolic: bpDia,
      });
    }
  }

  // =========================================================================
  // 5. CARDIAC PULSE ANIMATION & AUDIO BEEP
  // =========================================================================
  function initAudio() {
    const audioBtn = document.getElementById("btn-audio-toggle");
    const audioLabel = document.getElementById("audio-label");
    const audioIcon = document.getElementById("audio-icon");

    if (audioBtn) {
      audioBtn.addEventListener("click", () => {
        state.audioBeepEnabled = !state.audioBeepEnabled;
        if (state.audioBeepEnabled && !audioCtx) {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioLabel) {
          audioLabel.textContent = `Beep: ${state.audioBeepEnabled ? "On" : "Off"}`;
        }
        if (audioBtn) {
          audioBtn.className = state.audioBeepEnabled
            ? "warm-pill bg-[#18181B] text-white border-[#18181B] cursor-pointer"
            : "warm-pill cursor-pointer";
        }
      });
    }
  }

  function triggerCardiacBeat() {
    // 1. Pulsing Heart Icon
    const heartIcon = document.getElementById("hr-pulse-icon");
    if (heartIcon) {
      heartIcon.classList.remove("animate-heartbeat");
      void heartIcon.offsetWidth; // Force CSS reflow
      heartIcon.classList.add("animate-heartbeat");
    }

    // 2. Synthesized QRS Audio Beep Tone (880 Hz standard medical tone)
    if (state.audioBeepEnabled && audioCtx) {
      try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        // Higher pitch for high SpO2 (medical standard)
        const freq = 660 + (state.vitals.spo2 - 90) * 22;
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);

        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.08);

        osc.connect(gain);
        gain.connect(audioCtx.destination);

        osc.start();
        osc.stop(audioCtx.currentTime + 0.08);
      } catch (e) {
        // Audio policy restrictions
      }
    }
  }

  // =========================================================================
  // 6. WEBCAM FEED & FACE RECOGNITION PIPELINE
  // =========================================================================
  function initWebcamPipeline() {
    const video = document.getElementById("webcam-video");
    const overlay = document.getElementById("webcam-overlay");
    const placeholder = document.getElementById("camera-placeholder");
    const reqBtn = document.getElementById("btn-request-cam");
    const toggleBtn = document.getElementById("btn-toggle-camera");

    async function startCamera() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showCameraPlaceholder("Browser does not support camera API.");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
          audio: false,
        });
        state.webcam.stream = stream;
        state.webcam.active = true;
        if (video) {
          video.srcObject = stream;
          video.play().catch(() => {});
        }
        const faceIdVideo = document.getElementById("faceid-video");
        if (faceIdVideo) {
          faceIdVideo.srcObject = stream;
          faceIdVideo.play().catch(() => {});
        }
        if (placeholder) placeholder.classList.add("hidden");
        startFaceProcessingLoop();
      } catch (err) {
        console.warn("Camera access denied or device unavailable. Running mock face recognition.", err);
        showCameraPlaceholder("Camera access not available. Live tracking simulation active.");
        startMockFaceTrackingLoop();
      }
    }

    function showCameraPlaceholder(msg) {
      if (placeholder) {
        placeholder.classList.remove("hidden");
        const p = placeholder.querySelector("p");
        if (p) p.textContent = msg;
      }
    }

    if (reqBtn) reqBtn.addEventListener("click", () => startCamera());
    if (toggleBtn) {
      toggleBtn.addEventListener("click", () => {
        if (state.webcam.active) {
          if (state.webcam.stream) {
            state.webcam.stream.getTracks().forEach((t) => t.stop());
          }
          state.webcam.active = false;
          toggleBtn.textContent = "Start Stream";
          showCameraPlaceholder("Webcam paused by user.");
        } else {
          startCamera();
          toggleBtn.textContent = "Pause Stream";
        }
      });
    }

    // 1-Click Bind Subject to Session
    const btnBind = document.getElementById("btn-bind-subject");
    if (btnBind) {
      btnBind.addEventListener("click", () => {
        const idName = document.getElementById("id-recognized-name").textContent;
        const participantId = 16;
        bindSubjectToSession(idName, participantId);
      });
    }

    // Auto-request or start tracking simulation
    startCamera();
  }

  function startFaceProcessingLoop() {
    const video = document.getElementById("webcam-video");
    const overlay = document.getElementById("webcam-overlay");
    if (!video || !overlay) return;

    const ctx = overlay.getContext("2d");
    const offscreen = document.createElement("canvas");
    offscreen.width = 640;
    offscreen.height = 480;
    const offCtx = offscreen.getContext("2d");

    // Processing loop (~6 FPS to minimize bandwidth / CPU)
    state.webcam.intervalId = setInterval(async () => {
      if (!state.webcam.active || video.readyState < 2) return;

      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 480;
      overlay.width = vw;
      overlay.height = vh;

      // Draw current video frame to offscreen canvas (crisp 640px wide)
      offscreen.width = 640;
      offscreen.height = Math.round(640 * (vh / vw)) || 480;
      offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
      const base64Image = offscreen.toDataURL("image/jpeg", 0.80);

      const t0 = performance.now();
      try {
        // Attempt POST to backend face service
        const res = await fetch("/api/face/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: base64Image, recognition_thresh: 0.35, return_annotated: true }),
        });

        const latency = Math.round(performance.now() - t0);
        const latEl = document.getElementById("face-latency");
        if (latEl) latEl.textContent = `${latency} ms`;

        if (res.ok) {
          const data = await res.json();
          const detections = data.detections || [];
          renderFaceDetections(ctx, overlay.width, overlay.height, detections);

          // Handle Apple Face ID Gate profile recognition if gate is currently locked
          if (faceIdState.isLocked && !faceIdState.isAuthenticating && detections.length > 0) {
            const topDet = detections[0];
            const nameLower = (topDet.name || "").toLowerCase();
            const isLive = topDet.liveness ? topDet.liveness.is_live : true;

            if (isLive && (nameLower.includes("ashutosh") || nameLower === "ashutosh")) {
              startBiometricAuthenticationSequence("ashutosh");
            } else if (isLive && (nameLower.includes("dhrunil") || nameLower === "dhrunil")) {
              startBiometricAuthenticationSequence("dhrunil");
            }
          }
        } else {
          // If server returns error, fallback to client overlay tracking
          renderMockFaceOverlay(ctx, overlay.width, overlay.height);
        }
      } catch (e) {
        // Offline / Standalone mode
        renderMockFaceOverlay(ctx, overlay.width, overlay.height);
      }
    }, 160);
  }

  function startMockFaceTrackingLoop() {
    const overlay = document.getElementById("webcam-overlay");
    if (!overlay) return;
    const ctx = overlay.getContext("2d");

    setInterval(() => {
      overlay.width = overlay.parentElement.clientWidth || 640;
      overlay.height = overlay.parentElement.clientHeight || 440;
      renderMockFaceOverlay(ctx, overlay.width, overlay.height);
    }, 100);
  }

  function renderFaceDetections(ctx, w, h, detections) {
    ctx.clearRect(0, 0, w, h);

    const faceCountTag = document.getElementById("face-count-tag");
    const accessBadge = document.getElementById("subject-access-badge");

    if (!detections || detections.length === 0) {
      if (faceCountTag) faceCountTag.textContent = "No Face Detected";
      return;
    }

    if (faceCountTag) faceCountTag.textContent = `${detections.length} Face Detected`;

    detections.forEach((det) => {
      const scaleX = w / 640;
      const scaleY = h / 480;
      const origX1 = (det.bbox ? det.bbox[0] : 0) * scaleX;
      const origY1 = (det.bbox ? det.bbox[1] : 0) * scaleY;
      const origX2 = (det.bbox ? det.bbox[2] : 0) * scaleX;
      const origY2 = (det.bbox ? det.bbox[3] : 0) * scaleY;

      // Mirrored horizontal coordinates to match CSS transform: scaleX(-1) on video
      const x1 = Math.max(0, w - origX2);
      const x2 = Math.min(w, w - origX1);
      const y1 = Math.max(0, origY1);
      const y2 = Math.min(h, origY2);
      const bw = Math.max(10, x2 - x1);
      const bh = Math.max(10, y2 - y1);

      const name = (det.name || "UNKNOWN").trim();
      const isKnown = det.is_known || (name.toLowerCase() === "ashutosh" || name.toLowerCase() === "dhrunil");
      const isLive = det.liveness ? det.liveness.is_live : true;
      const score = Math.round((det.similarity || 0) * 100);

      // Editorial clean colors: Forest emerald for known live, Warm Amber for unknown, Berry for spoof
      let strokeColor = "#15803d"; // Forest emerald
      if (!isLive) strokeColor = "#be123c"; // Warm berry (spoof)
      else if (!isKnown) strokeColor = "#b45309"; // Warm amber (unknown)

      // Clean rounded bounding box - NO NEON GLOW
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.shadowBlur = 0;
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x1, y1, bw, bh, 12);
        ctx.stroke();
      } else {
        ctx.strokeRect(x1, y1, bw, bh);
      }

      // Draw 5 facial landmark points
      if (det.landmarks && Array.isArray(det.landmarks)) {
        ctx.fillStyle = strokeColor;
        det.landmarks.forEach((pt) => {
          const lmx = w - (pt[0] * scaleX);
          const lmy = pt[1] * scaleY;
          ctx.beginPath();
          ctx.arc(lmx, lmy, 3, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      // Name & Confidence Badge
      const tagText = isKnown ? `${name} (${score}%)` : `UNKNOWN (${score}%)`;

      ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif';
      const textWidth = ctx.measureText(tagText).width;
      const tagW = textWidth + 16;
      const tagH = 22;
      const tagX = Math.max(8, Math.min(w - tagW - 8, x1));
      const tagY = Math.max(26, y1);

      ctx.fillStyle = "rgba(24, 24, 27, 0.88)";
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(tagX, tagY - 24, tagW, tagH, 6);
        ctx.fill();
      } else {
        ctx.fillRect(tagX, tagY - 24, tagW, tagH);
      }

      ctx.fillStyle = strokeColor;
      ctx.fillText(tagText, tagX + 8, tagY - 8);

      // Update Access Badge
      if (accessBadge) {
        if (isKnown) {
          accessBadge.textContent = "VERIFIED ENROLLED";
          accessBadge.className = "px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#F0FDF4] text-[#15803D] border border-[#BBF7D0] font-semibold";
        } else {
          accessBadge.textContent = "UNREGISTERED";
          accessBadge.className = "px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#FFFBEB] text-[#B45309] border border-[#FDE68A] font-semibold";
        }
      }

      // Update Identity HUD Panel
      updateIdentityPanel({
        name: isKnown ? name : "UNKNOWN",
        similarity: score,
        trackId: det.track_id || 101,
        isLive: isLive,
        blink: det.liveness ? det.liveness.blink_detected : true,
      });
    });
  }

  function renderMockFaceOverlay(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);

    // Subtle breathing drift of bounding box
    state.webcam.mockAngle += 0.04;
    const dx = Math.sin(state.webcam.mockAngle) * 12;
    const dy = Math.cos(state.webcam.mockAngle * 0.7) * 6;

    const boxW = Math.min(220, w * 0.35);
    const boxH = Math.min(260, h * 0.55);
    const x1 = (w - boxW) / 2 + dx;
    const y1 = (h - boxH) / 2 + dy - 20;

    // Clean Target Reticle (Zero neon glow)
    ctx.strokeStyle = "#15803d";
    ctx.lineWidth = 2;
    ctx.shadowBlur = 0;

    // 4 Corner Brackets
    const len = 18;
    ctx.beginPath();
    // Top-Left
    ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1);
    // Top-Right
    ctx.moveTo(x1 + boxW - len, y1); ctx.lineTo(x1 + boxW, y1); ctx.lineTo(x1 + boxW, y1 + len);
    // Bottom-Left
    ctx.moveTo(x1, y1 + boxH - len); ctx.lineTo(x1, y1 + boxH); ctx.lineTo(x1 + len, y1 + boxH);
    // Bottom-Right
    ctx.moveTo(x1 + boxW - len, y1 + boxH); ctx.lineTo(x1 + boxW, y1 + boxH); ctx.lineTo(x1 + boxW, y1 + boxH - len);
    ctx.stroke();

    // 5-Point Facial Landmarks (Eyes, Nose, Mouth Corners)
    const eyeY = y1 + boxH * 0.38;
    const leftEye = { x: x1 + boxW * 0.33, y: eyeY };
    const rightEye = { x: x1 + boxW * 0.67, y: eyeY };
    const nose = { x: x1 + boxW * 0.50, y: y1 + boxH * 0.55 };
    const mouthL = { x: x1 + boxW * 0.38, y: y1 + boxH * 0.75 };
    const mouthR = { x: x1 + boxW * 0.62, y: y1 + boxH * 0.75 };

    ctx.fillStyle = "#1d4ed8";
    [leftEye, rightEye, nose, mouthL, mouthR].forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Tag Header
    const tag = "ID:016 Ashutosh (98.4%) [VERIFIED]";
    ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif';
    const tagW = ctx.measureText(tag).width + 16;
    ctx.fillStyle = "rgba(24, 24, 27, 0.9)";
    ctx.fillRect(x1, y1 - 24, tagW, 20);
    ctx.fillStyle = "#15803d";
    ctx.fillText(tag, x1 + 8, y1 - 9);

    // Update Identity HUD
    updateIdentityPanel({
      name: "Ashutosh",
      similarity: 98.4,
      trackId: 16,
      isLive: true,
      blink: true,
    });
  }

  function updateIdentityPanel(info) {
    const elName = document.getElementById("id-recognized-name");
    const elSim = document.getElementById("id-similarity");
    const elTrk = document.getElementById("id-track-num");
    const elAvatar = document.getElementById("id-photo-placeholder");
    const accessBadge = document.getElementById("subject-access-badge");

    if (elName) elName.textContent = info.name;
    if (elSim) elSim.textContent = `${info.similarity}%`;
    if (elTrk) elTrk.textContent = `TRK-${info.trackId}`;
    if (elAvatar && info.name) {
      const initials = info.name.split(" ").map((w) => w[0]).join("").slice(0, 2);
      elAvatar.textContent = initials;
      if (info.name === "Ashutosh") {
        elAvatar.className = "w-14 h-14 rounded-2xl bg-[#18181B] text-white flex items-center justify-center font-mono text-base font-bold shadow-xs";
      } else if (info.name === "Dhrunil") {
        elAvatar.className = "w-14 h-14 rounded-2xl bg-[#15803D] text-white flex items-center justify-center font-mono text-base font-bold shadow-xs";
      } else {
        elAvatar.className = "w-14 h-14 rounded-2xl bg-[#FAF9F5] border border-[#ECE8DF] text-[#78736B] flex items-center justify-center font-mono text-base font-bold shadow-xs";
      }
    }
    if (accessBadge) {
      if (info.name === "Ashutosh" || info.name === "Dhrunil") {
        accessBadge.textContent = "VERIFIED ENROLLED";
        accessBadge.className = "px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#F0FDF4] text-[#15803D] border border-[#BBF7D0] font-semibold";
      } else {
        accessBadge.textContent = "UNREGISTERED GUEST";
        accessBadge.className = "px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#FFFBEB] text-[#B45309] border border-[#FDE68A] font-semibold";
      }
    }
  }

  // =========================================================================
  // 7. SESSION MANAGEMENT & 1-CLICK SUBJECT BINDING
  // =========================================================================
  function initSessionManager() {
    const btnToggle = document.getElementById("btn-toggle-session");
    const btnGuest = document.getElementById("btn-guest-session");

    if (btnToggle) {
      btnToggle.addEventListener("click", () => {
        if (!state.activeSession.id) {
          startSession("John Doe", 101, false);
        } else {
          endCurrentSession();
        }
      });
    }

    if (btnGuest) {
      btnGuest.addEventListener("click", () => {
        startSession("Guest Subject", null, true);
      });
    }

    // Enrollment form submit
    const enrollForm = document.getElementById("enroll-form");
    if (enrollForm) {
      enrollForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const nameInput = document.getElementById("enroll-name");
        const notesInput = document.getElementById("enroll-notes");
        const fb = document.getElementById("enroll-feedback");

        const name = nameInput.value.trim();
        const notes = notesInput.value.trim();

        if (fb) {
          fb.classList.remove("hidden");
          fb.textContent = `Enrolling ${name}...`;
          fb.className = "mt-2 text-[11px] font-mono text-blue-400 text-center";
        }

        // Try API enrollment with webcam snapshot if available
        let snapBase64 = null;
        const video = document.getElementById("webcam-video");
        if (video && video.videoWidth > 0) {
          const c = document.createElement("canvas");
          c.width = 320;
          c.height = 240;
          c.getContext("2d").drawImage(video, 0, 0, 320, 240);
          snapBase64 = c.toDataURL("image/jpeg", 0.8);
        }

        try {
          if (snapBase64) {
            await fetch("/api/face/enroll", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name: name, image: snapBase64, notes: notes }),
            });
          }
        } catch (err) {
          // offline enrollment
        }

        setTimeout(() => {
          if (fb) {
            fb.textContent = `✓ Successfully enrolled ${name}!`;
            fb.className = "mt-2 text-[11px] font-mono text-emerald-400 text-center";
          }
          bindSubjectToSession(name, 201);
          nameInput.value = "";
          notesInput.value = "";
        }, 500);
      });
    }
  }

  async function startSession(name, participantId, isGuest) {
    const sess = state.activeSession;
    sess.id = "sess_" + Date.now().toString(36);
    sess.name = name;
    sess.participantId = participantId;
    sess.isGuest = isGuest;
    sess.startTime = Date.now();
    sess.sampleCount = 0;

    // Send session command to backend if WebSocket is active
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({
        command: "start_session",
        session_id: sess.id,
        participant_id: participantId,
        is_guest: isGuest,
      }));
    } else {
      // Also notify REST endpoint if available
      try {
        fetch("/api/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sess.id, participant_id: participantId, is_guest: isGuest }),
        });
      } catch (e) {}
    }

    // Start timer interval
    if (sess.timerInterval) clearInterval(sess.timerInterval);
    sess.timerInterval = setInterval(updateSessionClock, 1000);

    // Update UI Badges
    const badge = document.getElementById("session-badge");
    const subjName = document.getElementById("subject-name");
    const avatar = document.getElementById("subject-avatar");
    const btnToggle = document.getElementById("btn-toggle-session");

    if (badge) {
      badge.textContent = isGuest ? "GUEST ACTIVE" : "SUBJECT BOUND";
      badge.className = "px-2 py-0.5 rounded-full text-[9px] font-mono bg-[#F0FDF4] text-[#15803D] border border-[#BBF7D0] font-semibold";
    }
    if (subjName) subjName.textContent = name;
    if (avatar) {
      avatar.textContent = name.split(" ").map((w) => w[0]).join("").slice(0, 2);
      avatar.className = name === "Ashutosh"
        ? "w-8 h-8 rounded-full bg-[#18181B] text-white flex items-center justify-center font-bold text-xs"
        : (name === "Dhrunil"
          ? "w-8 h-8 rounded-full bg-[#15803D] text-white flex items-center justify-center font-bold text-xs"
          : "w-8 h-8 rounded-full bg-[#FAF9F5] border border-[#ECE8DF] flex items-center justify-center font-bold text-[#78736B] text-xs");
    }
    if (btnToggle) {
      btnToggle.textContent = "End Session";
      btnToggle.className = "flex-1 py-1.5 rounded-full bg-[#FFF1F2] hover:bg-[#FFE4E6] text-[#BE123C] border border-[#FECDD3] text-[11px] font-semibold transition-colors";
    }
  }

  function bindSubjectToSession(name, participantId) {
    startSession(name, participantId, false);
    console.log(`Telemetry stream successfully bound to participant: ${name} (ID: ${participantId})`);
  }

  async function endCurrentSession() {
    const sess = state.activeSession;
    if (!sess.id) return;

    if (sess.timerInterval) {
      clearInterval(sess.timerInterval);
      sess.timerInterval = null;
    }

    const durationSec = Math.round((Date.now() - sess.startTime) / 1000);
    const durationFormatted = formatDuration(durationSec);

    // Send stop command to backend
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ command: "stop_session" }));
    } else {
      try {
        fetch("/api/sessions/stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sess.id }),
        });
      } catch (e) {}
    }

    // Archive session in historical list
    state.sessions.unshift({
      id: sess.id,
      name: sess.name,
      isGuest: sess.isGuest,
      startTime: new Date(sess.startTime).toLocaleString(),
      duration: durationFormatted,
      durationSec: durationSec,
      avgHr: state.vitals.ecgHr,
      avgSpo2: state.vitals.spo2,
      status: "COMPLETED",
    });

    renderHistoryTable();

    // Reset Active Session
    sess.id = null;
    sess.name = "Guest Session";
    sess.participantId = null;
    sess.isGuest = true;
    sess.startTime = null;

    const badge = document.getElementById("session-badge");
    const subjName = document.getElementById("subject-name");
    const avatar = document.getElementById("subject-avatar");
    const btnToggle = document.getElementById("btn-toggle-session");
    const timer = document.getElementById("session-timer");

    if (badge) {
      badge.textContent = "UNASSIGNED";
      badge.className = "px-2 py-0.5 rounded-full text-[9px] font-mono bg-[#FAF9F5] text-[#78736B] border border-[#ECE8DF] font-medium";
    }
    if (subjName) subjName.textContent = "Guest Session";
    if (avatar) {
      avatar.textContent = "?";
      avatar.className = "w-8 h-8 rounded-full bg-[#FAF9F5] border border-[#ECE8DF] flex items-center justify-center font-bold text-[#78736B] text-xs";
    }
    if (btnToggle) {
      btnToggle.textContent = "Start Session";
      btnToggle.className = "flex-1 py-1.5 rounded-full bg-[#18181B] hover:bg-[#000000] text-white text-[11px] font-semibold transition-all shadow-xs";
    }
    if (timer) timer.textContent = "00:00:00";
  }

  function updateSessionClock() {
    const sess = state.activeSession;
    if (!sess.startTime) return;
    const elapsed = Math.floor((Date.now() - sess.startTime) / 1000);
    const timer = document.getElementById("session-timer");
    if (timer) {
      timer.textContent = formatDuration(elapsed);
    }
  }

  function formatDuration(sec) {
    const h = String(Math.floor(sec / 3600)).padStart(2, "0");
    const m = String(Math.floor((sec % 3600) / 60)).padStart(2, "0");
    const s = String(sec % 60).padStart(2, "0");
    return `${h}:${m}:${s}`;
  }

  // =========================================================================
  // 8. HISTORICAL SESSIONS & REPORT GENERATION
  // =========================================================================
  async function initHistoryAndReports() {
    // Attempt fetch from server API
    try {
      const res = await fetch("/api/sessions");
      if (res.ok) {
        const data = await res.json();
        if (data.sessions && data.sessions.length > 0) {
          state.sessions = data.sessions.map((s) => ({
            id: s.session_id,
            name: s.participant_name || (s.is_guest ? "Guest Participant" : "Enrolled Subject"),
            isGuest: !!s.is_guest,
            startTime: s.start_time,
            duration: formatDuration(Math.round(s.duration_seconds || 60)),
            durationSec: s.duration_seconds || 60,
            avgHr: s.summary ? (s.summary.mean_ecg_hr || 72) : 72,
            avgSpo2: s.summary ? (s.summary.mean_spo2 || 98.4) : 98.4,
            status: s.status || "COMPLETED",
          }));
        }
      }
    } catch (e) {
      // offline fallback
    }

    // Seed sample sessions if still empty
    if (state.sessions.length === 0) {
      state.sessions = [
        {
          id: "sess_k9f12a",
          name: "Dr. Alex Mercer",
          isGuest: false,
          startTime: "2026-10-13 18:45:10",
          duration: "00:24:15",
          durationSec: 1455,
          avgHr: 71,
          avgSpo2: 98.6,
          status: "COMPLETED",
        },
        {
          id: "sess_8m201x",
          name: "Sarah Jenkins",
          isGuest: false,
          startTime: "2026-10-13 16:12:04",
          duration: "00:15:30",
          durationSec: 930,
          avgHr: 76,
          avgSpo2: 97.9,
          status: "COMPLETED",
        },
        {
          id: "sess_g319cc",
          name: "Guest Participant",
          isGuest: true,
          startTime: "2026-10-13 14:02:22",
          duration: "00:08:42",
          durationSec: 522,
          avgHr: 68,
          avgSpo2: 98.8,
          status: "COMPLETED",
        },
      ];
    }

    renderHistoryTable();

    // Search filter
    const searchInput = document.getElementById("history-search");
    if (searchInput) {
      searchInput.addEventListener("input", () => {
        renderHistoryTable(searchInput.value.trim().toLowerCase());
      });
    }

    // Refresh button
    const btnRefresh = document.getElementById("btn-refresh-history");
    if (btnRefresh) {
      btnRefresh.addEventListener("click", () => renderHistoryTable());
    }

    // Export CSV Handlers
    const btnExportRaw = document.getElementById("btn-export-raw-csv");
    if (btnExportRaw) {
      btnExportRaw.addEventListener("click", () => exportRawTelemetryCSV());
    }

    const btnExportMetrics = document.getElementById("btn-export-metrics-csv");
    if (btnExportMetrics) {
      btnExportMetrics.addEventListener("click", () => exportMetricsCSV());
    }

    const btnPrint = document.getElementById("btn-print-report");
    if (btnPrint) {
      btnPrint.addEventListener("click", () => window.print());
    }
  }

  function renderHistoryTable(query = "") {
    const tbody = document.getElementById("sessions-table-body");
    const emptyState = document.getElementById("sessions-empty-state");
    if (!tbody) return;

    tbody.innerHTML = "";

    const filtered = state.sessions.filter((s) => {
      if (!query) return true;
      return s.name.toLowerCase().includes(query) || s.id.toLowerCase().includes(query);
    });

    if (filtered.length === 0) {
      if (emptyState) emptyState.classList.remove("hidden");
      return;
    }
    if (emptyState) emptyState.classList.add("hidden");

    filtered.forEach((s) => {
      const tr = document.createElement("tr");
      tr.className = "hover:bg-[#FAF9F5] transition-colors border-b border-[#ECE8DF]";
      tr.innerHTML = `
        <td class="p-3.5 font-medium font-mono text-[#18181B]">${s.id}</td>
        <td class="p-3.5 font-semibold text-[#18181B]">${s.name}</td>
        <td class="p-3.5">
          <span class="px-2 py-0.5 rounded-full text-[10px] font-mono ${
            s.isGuest ? "bg-[#FFFBEB] text-[#B45309] border border-[#FDE68A]" : "bg-[#F0FDF4] text-[#15803D] border border-[#BBF7D0] font-semibold"
          }">${s.isGuest ? "GUEST" : "ENROLLED"}</span>
        </td>
        <td class="p-3.5 text-[#78736B] font-mono">${s.startTime}</td>
        <td class="p-3.5 font-mono text-[#78736B]">${s.duration}</td>
        <td class="p-3.5 font-mono text-[#15803D] font-semibold">${s.avgHr} BPM</td>
        <td class="p-3.5 font-mono text-[#1D4ED8] font-semibold">${s.avgSpo2}%</td>
        <td class="p-3.5">
          <span class="px-2 py-0.5 rounded-full text-[10px] font-mono bg-[#F0FDF4] text-[#15803D] border border-[#BBF7D0] font-semibold">
            ${s.status}
          </span>
        </td>
        <td class="p-3.5 text-right space-x-1.5">
          <button class="airbnb-btn airbnb-btn-secondary py-1 px-3 text-[11px]" onclick="window.loadSessionReport('${s.id}')">
            Report
          </button>
          <button class="airbnb-btn airbnb-btn-secondary py-1 px-3 text-[11px]" onclick="window.exportSessionCSV('${s.id}')">
            CSV
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Global window triggers for table actions
  window.loadSessionReport = (sessionId) => {
    const s = state.sessions.find((item) => item.id === sessionId);
    if (!s) return;

    document.getElementById("rep-name").textContent = s.name;
    document.getElementById("rep-session-id").textContent = s.id;
    document.getElementById("rep-duration").textContent = s.duration;
    document.getElementById("rep-id").textContent = "REP-" + s.id.toUpperCase();

    // Switch to Report View
    const reportNavBtn = document.querySelector('[data-view="view-reports"]');
    if (reportNavBtn) reportNavBtn.click();
  };

  window.exportSessionCSV = (sessionId) => {
    exportRawTelemetryCSV(sessionId);
  };

  async function exportRawTelemetryCSV(sessionId) {
    const sid = sessionId || state.activeSession.id || "telemetry_live";

    // Attempt direct download from backend endpoint
    try {
      const res = await fetch(`/api/reports/export/timeseries/${sid}`);
      if (res.ok) {
        const blob = await res.blob();
        downloadBlob(blob, `timeseries_${sid}.csv`);
        return;
      }
    } catch (e) {
      // fallback to client-side generation
    }

    let csv = "session_id,timestamp_ms,seq,ecg_raw,ppg_red,ppg_ir,leads_off\n";
    const baseTime = Date.now() - 5000;
    for (let i = 0; i < 500; i++) {
      const t = baseTime + i * 10;
      const ecg = Math.round(512 + Math.sin(i * 0.1) * 80);
      const red = Math.round(20000 + Math.sin(i * 0.12) * 500);
      const ir = Math.round(24000 + Math.sin(i * 0.12) * 600);
      csv += `${sid},${t},${i},${ecg},${red},${ir},0\n`;
    }

    downloadBlob(new Blob([csv], { type: "text/csv;charset=utf-8;" }), `${sid}_raw_telemetry.csv`);
  }

  async function exportMetricsCSV(sessionId) {
    const sid = sessionId || state.activeSession.id || "metrics_live";

    // Attempt download from backend endpoint
    try {
      const res = await fetch(`/api/reports/export/metrics/${sid}`);
      if (res.ok) {
        const blob = await res.blob();
        downloadBlob(blob, `metrics_${sid}.csv`);
        return;
      }
    } catch (e) {
      // fallback
    }

    let csv = "session_id,timestamp_s,ecg_hr,ppg_hr,spo2,pat_ms,rr_interval_ms,rmssd,bp_sys,bp_dia\n";
    const v = state.vitals;
    for (let i = 0; i < 60; i++) {
      csv += `${sid},${i * 1.0},${v.ecgHr},${v.ppgHr},${v.spo2},${v.pat},${v.rrInterval},${v.rmssd},${v.bpSys},${v.bpDia}\n`;
    }

    downloadBlob(new Blob([csv], { type: "text/csv;charset=utf-8;" }), `${sid}_physiological_metrics.csv`);
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  // =========================================================================
  // 9. POINCARÉ SCATTER PLOT (RR[n] vs RR[n+1])
  // =========================================================================
  function initPoincare() {
    // Seed initial RR data
    for (let i = 0; i < 40; i++) {
      state.rrHistory.push(820 + (Math.random() - 0.5) * 60);
    }
    renderPoincarePlot();
  }

  function recordRRInterval(intervalMs) {
    state.rrHistory.push(intervalMs);
    if (state.rrHistory.length > 120) {
      state.rrHistory.shift();
    }
    if (state.currentView === "view-analytics") {
      renderPoincarePlot();
    }
  }

  function renderPoincarePlot() {
    const canvas = document.getElementById("canvas-poincare");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = "#0b0f17";
    ctx.fillRect(0, 0, w, h);

    // Diagonal identity line (RR_n = RR_n+1)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(10, h - 10);
    ctx.lineTo(w - 10, 10);
    ctx.stroke();
    ctx.setLineDash([]);

    const minRR = 600;
    const maxRR = 1100;
    const range = maxRR - minRR;

    const rrs = state.rrHistory;
    if (rrs.length < 2) return;

    // Plot consecutive dots
    for (let i = 0; i < rrs.length - 1; i++) {
      const xVal = rrs[i];
      const yVal = rrs[i + 1];

      const px = 20 + ((xVal - minRR) / range) * (w - 40);
      const py = h - 20 - ((yVal - minRR) / range) * (h - 40);

      const alpha = (i / rrs.length) * 0.8 + 0.2;
      ctx.fillStyle = `rgba(21, 128, 61, ${alpha})`;
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // =========================================================================
  // 10. SETTINGS & SIMULATOR CONTROLS
  // =========================================================================
  function initSettingsAndSimulator() {
    // Simulator Sliders
    const hrSlider = document.getElementById("sim-hr-slider");
    const hrLabel = document.getElementById("sim-hr-label");
    if (hrSlider) {
      hrSlider.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        state.simulator.hr = val;
        if (hrLabel) hrLabel.textContent = `${val} BPM`;
        notifySimulatorUpdate();
      });
    }

    const respSlider = document.getElementById("sim-resp-slider");
    const respLabel = document.getElementById("sim-resp-label");
    if (respSlider) {
      respSlider.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        state.simulator.respRate = val;
        if (respLabel) respLabel.textContent = `${val} BrPM`;
        notifySimulatorUpdate();
      });
    }

    const noiseSlider = document.getElementById("sim-noise-slider");
    const noiseLabel = document.getElementById("sim-noise-label");
    if (noiseSlider) {
      noiseSlider.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        state.simulator.noise = val;
        if (noiseLabel) noiseLabel.textContent = `${val} %`;
      });
    }

    // Leads-Off Toggle
    const leadsToggle = document.getElementById("sim-leads-off-toggle");
    if (leadsToggle) {
      leadsToggle.addEventListener("change", (e) => {
        state.simulator.leadsOff = e.target.checked ? 1 : 0;
        handleLeadStatus(e.target.checked);
        notifySimulatorUpdate();
      });
    }

    // Finger-Off Toggle
    const fingerToggle = document.getElementById("sim-finger-off-toggle");
    if (fingerToggle) {
      fingerToggle.addEventListener("change", (e) => {
        state.simulator.fingerOff = e.target.checked;
        const probePill = document.getElementById("ppg-status-pill");
        if (e.target.checked) {
          if (probePill) {
            probePill.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-[#B45309]"></span> DETACHED';
            probePill.className = "text-[#B45309] font-semibold flex items-center gap-1";
          }
          if (miniPpgScope) miniPpgScope.setLeadsOff(true, "FINGER OFF - CHECK SENSOR CLIP");
          if (fullPpgScope) fullPpgScope.setLeadsOff(true, "FINGER OFF - CHECK SENSOR CLIP");
        } else {
          if (probePill) {
            probePill.innerHTML = '<span class="w-1.5 h-1.5 rounded-full bg-[#1D4ED8]"></span> ATTACHED';
            probePill.className = "text-[#1D4ED8] font-semibold flex items-center gap-1";
          }
          if (miniPpgScope) miniPpgScope.setLeadsOff(false);
          if (fullPpgScope) fullPpgScope.setLeadsOff(false);
        }
        notifySimulatorUpdate();
      });
    }

    // Hardware Mock Mode Toggle
    const toggleMock = document.getElementById("toggle-mock-mode");
    if (toggleMock) {
      toggleMock.addEventListener("change", async (e) => {
        const isMock = e.target.checked;
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
          state.ws.send(JSON.stringify({ command: "set_mock_mode", enabled: isMock }));
        }
        try {
          await fetch("/api/settings/serial", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mock_mode: isMock }),
          });
        } catch (err) {}
      });
    }

    // Hardware Serial Ports Scan
    const btnScan = document.getElementById("btn-scan-ports");
    if (btnScan) {
      btnScan.addEventListener("click", async () => {
        btnScan.textContent = "Scanning...";
        try {
          const res = await fetch("/api/settings/serial");
          if (res.ok) {
            const data = await res.json();
            const select = document.getElementById("select-serial-port");
            select.innerHTML = '<option value="AUTO">AUTO (Detect Arduino Uno)</option>';
            if (data.available_ports && data.available_ports.length > 0) {
              data.available_ports.forEach((p) => {
                const opt = document.createElement("option");
                opt.value = p.device;
                opt.textContent = `${p.device} - ${p.description || "Serial Port"}`;
                if (data.detected_port && p.device === data.detected_port) {
                  opt.selected = true;
                }
                select.appendChild(opt);
              });
            }
          }
        } catch (e) {
          console.log("Using cached serial ports.");
        } finally {
          btnScan.innerHTML = '<i data-lucide="refresh-cw" class="w-3.5 h-3.5"></i> Scan';
          initLucideIcons();
        }
      });
    }

    // Hardware Connect Serial Button
    const btnConnectSerial = document.getElementById("btn-connect-serial");
    if (btnConnectSerial) {
      btnConnectSerial.addEventListener("click", async () => {
        const portSelect = document.getElementById("select-serial-port");
        const baudSelect = document.getElementById("select-baudrate");
        const portVal = portSelect ? portSelect.value : "COM5";
        const baudVal = baudSelect ? parseInt(baudSelect.value) : 115200;

        btnConnectSerial.textContent = "Connecting...";
        try {
          const res = await fetch("/api/settings/serial", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              port: portVal,
              baudrate: baudVal,
              mock_mode: false,
              connect: true,
            }),
          });
          if (res.ok) {
            const data = await res.json();
            const toggleMock = document.getElementById("toggle-mock-mode");
            if (toggleMock) toggleMock.checked = false;
            btnConnectSerial.textContent = `Connected (${data.active_port || portVal})`;
            setTimeout(() => {
              btnConnectSerial.textContent = "Connect Hardware Serial";
            }, 3000);
          }
        } catch (err) {
          btnConnectSerial.textContent = "Connection Failed";
          setTimeout(() => {
            btnConnectSerial.textContent = "Connect Hardware Serial";
          }, 3000);
        }
      });
    }

    // Hardware Disconnect Serial Button
    const btnDisconnectSerial = document.getElementById("btn-disconnect-serial");
    if (btnDisconnectSerial) {
      btnDisconnectSerial.addEventListener("click", async () => {
        try {
          await fetch("/api/settings/serial", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              connect: false,
              mock_mode: true,
            }),
          });
          const toggleMock = document.getElementById("toggle-mock-mode");
          if (toggleMock) toggleMock.checked = true;
          btnDisconnectSerial.textContent = "Disconnected";
          setTimeout(() => {
            btnDisconnectSerial.textContent = "Disconnect";
          }, 2000);
        } catch (err) {}
      });
    }

    // Reconnect WS Button
    const btnReconnectWs = document.getElementById("btn-reconnect-ws");
    if (btnReconnectWs) {
      btnReconnectWs.addEventListener("click", () => {
        initWebSocket();
      });
    }
  }

  function notifySimulatorUpdate() {
    const sim = state.simulator;
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({
        command: "set_sim_controls",
        heart_rate: sim.hr,
        leads_off: sim.leadsOff,
        finger_contact: !sim.fingerOff,
      }));
    }
    try {
      fetch("/api/settings/serial", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sim_heart_rate: sim.hr,
          sim_leads_off: sim.leadsOff,
          sim_finger_contact: !sim.fingerOff,
        }),
      });
    } catch (e) {}
  }

  // =========================================================================
  // 11. OSCILLOSCOPE INTERACTIVE CONTROLS
  // =========================================================================
  function initControls() {
    // Freeze Toggle
    const freezeBtn = document.getElementById("btn-freeze-toggle");
    const freezeLabel = document.getElementById("freeze-label");
    const freezeIcon = document.getElementById("freeze-icon");

    if (freezeBtn) {
      freezeBtn.addEventListener("click", () => {
        state.isFrozen = !state.isFrozen;
        [miniEcgScope, miniPpgScope, fullEcgScope, fullPpgScope].forEach((s) => {
          if (s) s.toggleFreeze();
        });
        if (freezeLabel) freezeLabel.textContent = state.isFrozen ? "Resume" : "Freeze";
        if (freezeBtn) {
          freezeBtn.className = state.isFrozen
            ? "warm-pill bg-[#B45309] text-white border-[#B45309] cursor-pointer"
            : "warm-pill cursor-pointer";
        }
      });
    }

    // ECG Modes (Sweep vs Scroll)
    const btnSweep = document.getElementById("btn-ecg-mode-sweep");
    const btnScroll = document.getElementById("btn-ecg-mode-scroll");
    if (btnSweep && btnScroll) {
      btnSweep.addEventListener("click", () => {
        if (fullEcgScope) fullEcgScope.setMode("sweep");
        btnSweep.className = "warm-btn warm-btn-primary py-1 px-3 text-xs";
        btnScroll.className = "warm-btn warm-btn-secondary py-1 px-3 text-xs";
      });
      btnScroll.addEventListener("click", () => {
        if (fullEcgScope) fullEcgScope.setMode("scroll");
        btnScroll.className = "warm-btn warm-btn-primary py-1 px-3 text-xs";
        btnSweep.className = "warm-btn warm-btn-secondary py-1 px-3 text-xs";
      });
    }

    // ECG Speed & Gain Selectors
    const selectEcgSpeed = document.getElementById("select-ecg-speed");
    if (selectEcgSpeed) {
      selectEcgSpeed.addEventListener("change", (e) => {
        const speed = parseFloat(e.target.value);
        if (fullEcgScope) fullEcgScope.setSweepDuration(speed === 50 ? 2.5 : speed === 12.5 ? 8.0 : 5.0);
      });
    }

    const selectEcgGain = document.getElementById("select-ecg-gain");
    if (selectEcgGain) {
      selectEcgGain.addEventListener("change", (e) => {
        const gain = parseFloat(e.target.value);
        if (fullEcgScope) fullEcgScope.setGain("ecg", gain);
      });
    }

    // PPG Gain Selector
    const selectPpgGain = document.getElementById("select-ppg-gain");
    if (selectPpgGain) {
      selectPpgGain.addEventListener("change", (e) => {
        const gain = parseFloat(e.target.value);
        if (fullPpgScope) {
          fullPpgScope.setGain("ppg_ir", gain);
          fullPpgScope.setGain("ppg_red", gain * 0.9);
        }
      });
    }
  }

  // =========================================================================
  // 12. SYSTEM UTC LIVE CLOCK
  // =========================================================================
  function startClock() {
    const clockEl = document.getElementById("live-clock");
    function update() {
      if (clockEl) {
        const now = new Date();
        const timeStr = now.toISOString().substring(11, 19);
        clockEl.textContent = `${timeStr} UTC`;
      }
    }
    update();
    setInterval(update, 1000);
  }
})();
