interface Props {
  value: number;
  onChange: (v: number) => void;
}

export default function LookbackSlider({ value, onChange }: Props) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-sm text-gray-400 whitespace-nowrap">回溯场数</span>
      <input
        type="range"
        min={5}
        max={15}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-32 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                   accent-blue-600"
      />
      <span className="text-sm font-mono text-gray-200 w-8">{value}</span>
      <span className="text-xs text-gray-500">场</span>
    </div>
  );
}
