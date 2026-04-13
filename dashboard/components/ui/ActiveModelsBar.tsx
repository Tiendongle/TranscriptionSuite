import React from 'react';
import { Activity, Languages, Server, Mic } from 'lucide-react';
import { useServerStatus } from '../../src/hooks/useServerStatus';
import { StatusLight } from './StatusLight';

export const ActiveModelsBar: React.FC = () => {
  const { details, serverStatus, ready } = useServerStatus();

  // Defensive checks since details might be null initially
  const models = (details?.models as any) || {};
  const isServerRunning = serverStatus !== 'inactive' && serverStatus !== 'error';

  // Extract model states
  const transcriptionStatus = models.transcription || {};
  const isMainLoaded = transcriptionStatus.loaded === true;
  const isMainDisabled = transcriptionStatus.disabled === true;
  const mainModelName = transcriptionStatus.selected_model || 'None';

  // Live model: If not explicitly provided by the new get_status update, fallback to None
  // (We'll update model_manager.py next to feed this explicit field)
  const isLiveLoaded = models.realtime?.active_sessions > 0 || isMainLoaded; // Approximated until backend update
  const liveModelName = models.realtime?.selected_model || models.transcription?.selected_model || 'None';
  const isLiveDisabled = !models.realtime?.selected_model && isMainDisabled;

  const translationStatus = models.translation || {};
  const isTranslationLoaded = translationStatus.loaded === true;
  const translationModelName = translationStatus.selected_model || 'None';

  const diarizationStatus = models.diarization || {};
  const isDiarizationLoaded = diarizationStatus.loaded === true;

  const renderBadge = (
    title: string,
    modelName: string,
    isLoaded: boolean,
    isDisabled: boolean,
    icon: React.ReactNode,
    colorClass: string
  ) => {
    // If server is entirely down, everything is offline
    if (!isServerRunning) {
      return (
        <div className={`flex flex-col space-y-1 rounded-xl border border-white/5 bg-white/5 p-3 shadow-sm opacity-60`}>
          <div className="flex items-center gap-2">
            <div className={`bg-slate-500/20 text-slate-500 rounded p-1.5`}>
              {icon}
            </div>
            <span className="text-xs font-semibold tracking-wide text-slate-400">{title}</span>
          </div>
          <span className="text-xs text-slate-600 truncate pl-8">Server Offline</span>
        </div>
      );
    }

    if (isDisabled) {
      return (
        <div className={`flex flex-col space-y-1 rounded-xl border border-white/5 bg-white/5 p-3 shadow-sm opacity-60`}>
          <div className="flex items-center gap-2">
            <div className={`bg-slate-500/20 text-slate-500 rounded p-1.5`}>
              {icon}
            </div>
            <span className="text-xs font-semibold tracking-wide text-slate-400">{title}</span>
          </div>
          <span className="text-xs text-amber-500/70 truncate pl-8">Disabled</span>
        </div>
      );
    }

    return (
      <div className={`flex flex-col space-y-1 rounded-xl border border-white/5 bg-white/5 p-3 shadow-sm transition-all duration-300 ${isLoaded ? `border-[${colorClass.split('-')[1]}]/20 bg-[${colorClass.split('-')[1]}]/5 shadow-[0_0_10px_rgba(var(--tw-colors-${colorClass.split('-')[1]}-400),0.05)]` : ''}`}>
         <div className="flex items-center justify-between">
           <div className="flex items-center gap-2">
              <div className={`${isLoaded ? `bg-${colorClass}/20` : 'bg-slate-500/20'} ${isLoaded ? `text-${colorClass}` : 'text-slate-400'} rounded p-1.5 transition-colors`}>
                {icon}
              </div>
              <span className="text-xs font-semibold tracking-wide text-white">{title}</span>
           </div>
           <StatusLight status={isLoaded ? 'active' : (ready ? 'inactive' : 'warning')} className="h-2 w-2" />
         </div>
         <span className={`text-xs ${isLoaded ? 'text-slate-300' : 'text-slate-500'} truncate pl-8 font-mono`} title={modelName}>
           {modelName}
         </span>
      </div>
    );
  };

  return (
    <div className="w-full">
      <div className="mb-2 px-1 flex items-center justify-between">
         <span className="text-xs font-semibold tracking-wider text-slate-400 uppercase">Active Models Source of Truth</span>
         {!isServerRunning && (
             <span className="text-xs text-red-400">Not Connected</span>
         )}
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {renderBadge(
          "Main ASR",
          mainModelName,
          isMainLoaded,
          isMainDisabled,
          <Server size={14} />,
          "accent-magenta"
        )}
        {renderBadge(
          "Live ASR",
          liveModelName,
          isLiveLoaded,
          isLiveDisabled,
          <Mic size={14} />,
          "accent-cyan"
        )}
         {renderBadge(
          "Diarization",
           isDiarizationLoaded ? "Speaker IDs Active" : "Inactive",
          isDiarizationLoaded,
          false,
          <Activity size={14} />,
          "emerald-400"
        )}
        {renderBadge(
          "Translation",
          translationModelName,
          isTranslationLoaded,
          translationModelName === 'None',
          <Languages size={14} />,
          "blue-400"
        )}
      </div>
    </div>
  );
};
