import * as React from "react";
import { cn } from "@/lib/utils";

type ContainerSize = "sm" | "md" | "lg" | "xl" | "full";

const sizeClasses: Record<ContainerSize, string> = {
  sm: "max-w-2xl",
  md: "max-w-4xl",
  lg: "max-w-6xl",
  xl: "max-w-7xl",
  full: "max-w-full",
};

interface ContainerProps extends React.HTMLAttributes<HTMLElement> {
  as?: React.ElementType;
  size?: ContainerSize;
  className?: string;
  children: React.ReactNode;
}

const Container = React.forwardRef<HTMLElement, ContainerProps>(
  (
    { as: Tag = "div", size = "xl", className, children, ...props },
    ref
  ) => {
    return (
      <Tag
        ref={ref}
        className={cn(
          "mx-auto w-full px-4 sm:px-6 lg:px-8",
          sizeClasses[size],
          className
        )}
        {...props}
      >
        {children}
      </Tag>
    );
  }
);

Container.displayName = "Container";

export { Container, type ContainerProps, type ContainerSize };
