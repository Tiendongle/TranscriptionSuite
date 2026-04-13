import React, { useCallback, useEffect, useRef, useState } from 'react';

interface LogEntry {
  timestamp: string;
  source: string;
  message: string;
  type?: 'info' | 'error' | 'success' | 'warning';
}

interface LogTerminalProps {
  title: string;
  logs: LogEntry[];
  className?: string;
  color?: 'cyan' | 'magenta' | 'orange';
}

// Cap the number of rendered rows to keep the GPU compositor within budget.
// Rendering 2,000 rows with hover transitions in one paint overflows Chromium's
// command buffer and triggers "GPU state invalid after WaitForGetOffsetInRange".
const MAX_VISIBLE_LINES = 200;

export const LogTerminal: React.FC<LogTerminalProps> = ({
  title,
  logs,
  className = '',
  color = 'cyan',
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Show only the most recent MAX_VISIBLE_LINES entries
  const visibleLogs = logs.length > MAX_VISIBLE_LINES ? logs.slice(-MAX_VISIBLE_LINES) : logs;
  const hiddenCount = logs.length - visibleLogs.length;

  // Pause auto-scroll when the user scrolls inside the terminal body
  const handleWheel = useCallback(() => {
    setAutoScroll(false);
  }, []);

  // Re-enable auto-scroll when the cursor leaves the terminal body
  const handleMouseLeave = useCallback(() => {
    setAutoScroll(true);
    // Immediately snap to bottom on re-enable
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const accentColor = {
    cyan: 'text-accent-cyan border-accent-cyan/20',
    magenta: 'text-accent-magenta border-accent-magenta/20',
    orange: 'text-accent-orange border-accent-orange/20',
  }[color];

  return (
    <div
      className={`flex flex-col overflow-hidden rounded-xl border border-white/10 bg-[#0b1120] shadow-2xl ${className}`}
    >
      {/* Terminal Header */}
      <div className="flex items-center justify-between border-b border-white/5 bg-white/5 px-4 py-2">
        <div className="flex items-center gap-2">
          <span
            className={`font-mono text-xs font-bold tracking-wider uppercase ${accentColor.split(' ')[0]}`}
          >
            &gt;_ {title}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="animate-pulse font-mono text-[10px] text-green-400">LIVE</span>
        </div>
      </div>

      {/* Terminal Body */}
      <div
        ref={scrollRef}
        onWheel={handleWheel}
        onMouseLeave={handleMouseLeave}
        className="custom-scrollbar selectable-text flex-1 space-y-1.5 overflow-y-auto p-4 font-mono text-xs"
      >
        {hiddenCount > 0 && (
          <div className="text-slate-600 italic select-none mb-2">
            ↑ {hiddenCount} earlier {hiddenCount === 1 ? 'entry' : 'entries'} not shown
          </div>
        )}
        {visibleLogs.map((log, index) => (
          <div key={index} className="flex gap-3 rounded p-0.5 transition-colors hover:bg-white/5">
            <span className="shrink-0 text-slate-600 select-none">{log.timestamp}</span>
            <span
              className={`break-all ${
                log.type === 'error'
                  ? 'text-red-400'
                  : log.type === 'success'
                    ? 'text-green-400'
                    : log.type === 'warning'
                      ? 'text-orange-400'
                      : 'text-slate-300'
              }`}
            >
              {log.message}
            </span>
          </div>
        ))}
        {logs.length === 0 && (
          <div className="text-slate-700 italic select-none">Waiting for stream...</div>
        )}
      </div>
    </div>
  );
};
