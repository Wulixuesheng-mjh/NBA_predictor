import { useState, useRef, useEffect } from "react";
import type { Team } from "../../types";

interface Props {
  teams: Team[];
  value: number | null;
  onChange: (id: number | null) => void;
  placeholder?: string;
}

export default function TeamSelector({
  teams,
  value,
  onChange,
  placeholder = "选择球队...",
}: Props) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Focus search input when dropdown opens
  useEffect(() => {
    if (open) {
      setTimeout(() => searchInputRef.current?.focus(), 50);
    } else {
      setSearch("");
    }
  }, [open]);

  const selected = teams.find((t) => t.id === value);

  const filtered = teams.filter((t) => {
    if (!search) return true;
    const s = search.toLowerCase();
    return (
      t.full_name.toLowerCase().includes(s) ||
      t.abbreviation.toLowerCase().includes(s)
    );
  });

  return (
    <div ref={containerRef} className="relative">
      {/* Dropdown trigger button */}
      <button
        type="button"
        className={`w-full bg-gray-800 border rounded-lg px-4 py-3 text-left
                   focus:outline-none focus:ring-2 focus:ring-blue-500
                   hover:border-gray-600 transition-colors flex items-center justify-between
                   ${open ? "border-blue-500 ring-2 ring-blue-500" : "border-gray-700"}`}
        onClick={() => setOpen(!open)}
      >
        <span className={selected ? "text-gray-100" : "text-gray-500"}>
          {selected
            ? `${selected.full_name} (${selected.abbreviation})`
            : placeholder}
        </span>
        <svg
          className={`w-4 h-4 text-gray-500 transition-transform duration-200 ${
            open ? "rotate-180" : ""
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {/* Dropdown panel */}
      {open && (
        <div className="absolute top-full mt-1 w-full bg-gray-800 border border-gray-700 rounded-lg
                        z-50 shadow-xl overflow-hidden">
          {/* Search input inside dropdown */}
          <div className="p-2 border-b border-gray-700">
            <input
              ref={searchInputRef}
              type="text"
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-sm
                         text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
              placeholder="输入城市名或队名筛选..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onClick={(e) => e.stopPropagation()}
            />
          </div>

          {/* Team list */}
          <div className="max-h-56 overflow-y-auto">
            {filtered.map((team) => (
              <button
                key={team.id}
                className={`w-full text-left px-4 py-2.5 hover:bg-gray-700 text-gray-200
                           flex items-center justify-between text-sm transition-colors
                           ${team.id === value ? "bg-blue-500/20 text-blue-300" : ""}`}
                onClick={() => {
                  onChange(team.id);
                  setOpen(false);
                }}
              >
                <span>{team.full_name}</span>
                <span className="text-gray-500 text-xs font-mono">
                  {team.abbreviation}
                </span>
              </button>
            ))}

            {filtered.length === 0 && (
              <div className="px-4 py-6 text-center text-sm text-gray-500">
                未找到匹配球队
              </div>
            )}
          </div>

          {/* Quick clear button */}
          {value !== null && (
            <div className="border-t border-gray-700 p-1">
              <button
                className="w-full text-center px-3 py-2 text-sm text-gray-500 hover:text-red-400
                           hover:bg-gray-700/50 rounded transition-colors"
                onClick={() => {
                  onChange(null);
                  setOpen(false);
                }}
              >
                清除选择
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
