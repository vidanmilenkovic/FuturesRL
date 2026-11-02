import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

function Svg({ children, ...props }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      width={16}
      height={16}
      {...props}
    >
      {children}
    </svg>
  );
}

export function DashboardIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <rect x="2.5" y="2.5" width="6" height="6" rx="1.2" />
      <rect x="11.5" y="2.5" width="6" height="6" rx="1.2" />
      <rect x="2.5" y="11.5" width="6" height="6" rx="1.2" />
      <rect x="11.5" y="11.5" width="6" height="6" rx="1.2" />
    </Svg>
  );
}

export function PipelineIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M3 4h6v3H3zM3 8.5h9v3H3zM3 13h12v3H3z" />
      <path d="M14 5.5h3M17 10h.01" />
    </Svg>
  );
}

export function RiskIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M10 2.5 3.5 5v4.6c0 3.6 2.6 6.6 6.5 7.9 3.9-1.3 6.5-4.3 6.5-7.9V5L10 2.5Z" />
      <path d="M10 7v3.2M10 13h.01" />
    </Svg>
  );
}

export function TermStructureIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M2.5 16.5h15" />
      <path d="M2.5 16.5V4" />
      <path d="M4.5 12.5 8 9l3 2.5 4.5-6" />
      <circle cx="8" cy="9" r="0.9" />
      <circle cx="11" cy="11.5" r="0.9" />
      <circle cx="15.5" cy="5.5" r="0.9" />
    </Svg>
  );
}

export function BacktestIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M2.5 16.5h15" />
      <path d="M3.5 16.5V3.5" />
      <path d="M5 13.5 8.5 10l2.5 1.8L16.5 5" />
    </Svg>
  );
}

export function ChevronDownIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="m5 8 5 5 5-5" />
    </Svg>
  );
}

export function ChevronUpIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="m5 12 5-5 5 5" />
    </Svg>
  );
}

export function RefreshIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M16.5 9.5a6.5 6.5 0 1 0-1.9 4.6" />
      <path d="M16.5 4.5v5h-5" />
    </Svg>
  );
}

export function PowerIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M10 2.5v6" />
      <path d="M5.6 5.4a6.5 6.5 0 1 0 8.8 0" />
    </Svg>
  );
}

export function ArrowRightIcon(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M4 10h12M11.5 5.5 16 10l-4.5 4.5" />
    </Svg>
  );
}
